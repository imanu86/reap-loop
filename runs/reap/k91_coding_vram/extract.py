import re
for cfg in ['full','k50','k91','k96']:
    raw = open(f'sites/site_{cfg}.log','rb').read().decode('utf-8','replace').replace('\r','\n')
    out=[]
    for ln in raw.split('\n'):
        if ln.startswith('ds4:') or 'prefill layer' in ln or ln.startswith('=== gen'): continue
        ln = re.sub(r'ds4: CUDA loading model tensors [0-9.]+ GiB cached','',ln)
        ln = re.sub(r'ds4: SPEX stats:.*','',ln)
        out.append(ln)
    text='\n'.join(out).strip()
    open(f'sites/{cfg}.html','w',encoding='utf-8').write(text)
    low=text.lower()
    def has(*keys): return any(k in low for k in keys)
    print(f"{cfg:5} chars={len(text):5} doctype={('<!doctype' in low)!s:5} html={('<html' in low)!s:5} "
          f"nav={has('<nav','navbar','nav-link')!s:5} hero={('hero' in low)!s:5} "
          f"form={('<form' in low)!s:5} input={low.count('<input')} script={('<script' in low)!s:5} "
          f"themeToggle={(has('theme') and has('toggle','dark'))!s:5} closes_html={('</html>' in low)!s:5} "
          f"index_html_degen={low.count('index_html')}")
