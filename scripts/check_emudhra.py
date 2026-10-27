"""Read-only pre-update TLS probes; no auth, no redirects, verification always enabled."""
import argparse, concurrent.futures, datetime, hashlib, importlib.util, json, pathlib, re, socket, ssl, sys, time, urllib.request, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-chains', type=pathlib.Path, required=True)
args = parser.parse_args()
HOSTS = [kind+'-'+gen+suffix+'-emudhra.we-have-a-problem.com'
         for kind in ('dv','ov','ev') for gen in ('g3b','g3c') for suffix in ('','-aaa')]
HOSTS += ['dv-g3b-noct-emudhra.we-have-a-problem.com', 'dv-g3b-aaa-noct-emudhra.we-have-a-problem.com']
assert len(HOSTS) == 14 and len(set(HOSTS)) == 14
REFERENCE = {}
with zipfile.ZipFile(args.reference_chains) as z:
    for host in HOSTS:
        blocks = re.findall(r'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', z.read('chains/'+host+'.pem').decode(), re.S)
        REFERENCE[host] = [hashlib.sha256(ssl.PEM_cert_to_DER_cert(p)).hexdigest() for p in blocks]

def fingerprint(sock):
    if sock is None:
        return {}
    out = {'leaf_sha256':hashlib.sha256(sock.getpeercert(True)).hexdigest()}
    chain_method = getattr(sock,'get_unverified_chain',None) or getattr(getattr(sock,'_sslobj',None),'get_unverified_chain',None)
    if chain_method:
        chain=[]
        for cert in chain_method():
            raw=cert if isinstance(cert,(bytes,str)) else cert.public_bytes()
            if isinstance(raw,str): raw=ssl.PEM_cert_to_DER_cert(raw)
            elif raw.startswith(b'-----BEGIN'): raw=ssl.PEM_cert_to_DER_cert(raw.decode())
            chain.append(hashlib.sha256(raw).hexdigest())
        out['chain_sha256']=chain
    return out

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

def probe(route,host):
    started=time.monotonic(); result={'route':route,'host':host,'control':host=='julialang.org'}
    url='https://'+host+'/'
    try:
        if route=='stdlib-ssl':
            with socket.create_connection((host,443),timeout=8) as raw:
                with ssl.create_default_context().wrap_socket(raw,server_hostname=host) as sock:
                    result.update(fingerprint(sock)); result['tls']=sock.version()
                    sock.sendall(('GET / HTTP/1.1\r\nHost: '+host+'\r\nUser-Agent: local-llm-server-tls-audit\r\nConnection: close\r\n\r\n').encode())
                    result['http_line']=sock.recv(1024).split(b'\r\n')[0].decode(errors='replace')
        elif route=='stdlib-urllib':
            # No cookie/auth handlers and no redirect following.
            opener=urllib.request.build_opener(NoRedirect)
            with opener.open(urllib.request.Request(url,headers={'User-Agent':'local-llm-server-tls-audit'}),timeout=8) as resp:
                result['http_status']=resp.status; result.update(fingerprint(resp.fp.raw._sock))
        elif route=='app-pinned-https':
            sys.path.insert(0,str(REPO)) if str(REPO) not in sys.path else None
            from local_llm_server import net_safety as net
            parsed=net._parsed_url(url); addresses=net.resolve_public_addresses(parsed)
            conn=net._connection(parsed,addresses[0],8)
            try:
                conn.request('GET','/',headers={'User-Agent':'local-llm-server-tls-audit','Connection':'close'})
                result.update(fingerprint(conn.sock)); resp=conn.getresponse(); result['http_status']=resp.status
            finally: conn.close()
        elif route in ('requests','pip-default'):
            if route=='requests':
                import requests
                sess=requests.Session()
            else:
                from pip._internal.cli.index_command import _create_truststore_ssl_context
                from pip._internal.network.session import PipSession
                sess=PipSession(ssl_context=_create_truststore_ssl_context())
            sess.trust_env=False # No implicit .netrc authentication; inventory found no proxies/CA overrides.
            with sess:
                with sess.get(url,timeout=8,allow_redirects=False,stream=True,headers={'User-Agent':'local-llm-server-tls-audit'}) as resp:
                    result['http_status']=resp.status_code
                    sock=getattr(getattr(resp.raw,'_connection',None),'sock',None)
                    if sock is None:
                        sock=getattr(getattr(getattr(getattr(resp.raw,'_fp',None),'fp',None),'raw',None),'_sock',None)
                    result.update(fingerprint(sock))
        elif route=='huggingface-client':
            from huggingface_hub.utils._http import default_client_factory
            # This factory does not attach Hub auth; do not call hf_hub_download or HfApi.
            with default_client_factory() as client:
                with client.stream('GET',url,follow_redirects=False,timeout=8,headers={'User-Agent':'local-llm-server-tls-audit'}) as resp:
                    result['http_status']=resp.status_code
                    stream=resp.extensions.get('network_stream')
                    result.update(fingerprint(stream.get_extra_info('ssl_object') if stream else None))
        elif route=='aiohttp':
            import asyncio, aiohttp
            async def request():
                async with aiohttp.ClientSession(trust_env=False) as client:
                    async with client.get(url, allow_redirects=False, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                        result['http_status']=resp.status
                        conn=resp.connection
                        sock=conn.transport.get_extra_info('ssl_object') if conn else None
                        result.update(fingerprint(sock))
            asyncio.run(request())
        elif route=='httpx':
            import httpx
            with httpx.Client(trust_env=False,follow_redirects=False,timeout=8) as client:
                with client.stream('GET',url) as resp:
                    result['http_status']=resp.status_code
                    stream=resp.extensions.get('network_stream')
                    result.update(fingerprint(stream.get_extra_info('ssl_object') if stream else None))
        result['connection']='established'
        expected=REFERENCE.get(host)
        if expected:
            result['reference_leaf_matches']=result.get('leaf_sha256')==expected[0]
            result['reference_intermediate_present']=expected[1] in result.get('chain_sha256',[])
            result['assessment']='accepted-confirmed' if result['reference_leaf_matches'] and result['reference_intermediate_present'] else 'unconfirmed-identity'
        else: result['assessment']='control-connected'
    except Exception as exc:
        errors=[]; seen=set(); queue=[exc]
        while queue:
            e=queue.pop(0)
            if id(e) in seen: continue
            seen.add(id(e)); errors.append({'type':type(e).__name__,'message':str(e),'verify_code':getattr(e,'verify_code',None),'verify_message':getattr(e,'verify_message',None)})
            queue += [v for v in [getattr(e,'__cause__',None),getattr(e,'__context__',None),getattr(e,'reason',None),*getattr(e,'args',[])] if isinstance(v,BaseException)]
        result['errors']=errors
        msg=' '.join(str(e) for e in errors).lower()
        if any(s in msg for s in ['expired','not yet valid','hostname mismatch','not valid for','certificate transparency','ct required']):
            result['assessment']='other-certificate-rejection'
        elif any(s in msg for s in ['unable to get local issuer','unable to verify','certificate revoked','untrusted','not trusted','compromised','certificate chain is not trusted','invalid issuer','certificate authority','certificate is blocked','certificate rejected']):
            result['assessment']='rejected-certificate'
        else: result['assessment']='unconfirmed-error'
    result['elapsed_seconds']=round(time.monotonic()-started,3)
    return result

routes=['stdlib-ssl','stdlib-urllib','app-pinned-https','requests','huggingface-client']
if importlib.util.find_spec('pip'): routes.append('pip-default')
if importlib.util.find_spec('httpx'): routes.append('httpx')
if importlib.util.find_spec('aiohttp'): routes.append('aiohttp')
output={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,'spec':{'control':'https://julialang.org/','no_auth':True,'follow_redirects':False,'tls_verification':True,'reference':'https://emudhra.we-have-a-problem.com/chains.zip'},'reference':REFERENCE,'results':[]}
for route in routes:
    control=probe(route,'julialang.org'); output['results'].append(control)
    if control['assessment']!='control-connected':
        output['results'] += [{'route':route,'host':h,'assessment':'unconfirmed-control-failed'} for h in HOSTS]
        continue
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        output['results'] += list(pool.map(lambda h:probe(route,h),HOSTS))
print(json.dumps(output,ensure_ascii=False,indent=2))
