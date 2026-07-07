def clean(cfg):
    t=open(f'sites/{cfg}.html',encoding='utf-8').read()
    i=t.lower().find('<!doctype')
    if i<0: i=t.lower().find('<html')
    if i>=0: t=t[i:]
    t=t.split('```')[0].rstrip()
    low=t.lower()
    if '</html>' not in low:
        if '<script' in low and '</script>' not in low: t+='\n</script>'
        if '</body>' not in low: t+='\n</body>'
        t+='\n</html>'
    open(f'sites/{cfg}_render.html','w',encoding='utf-8').write(t)
    print(f"{cfg}_render.html : {len(t)} chars, closed for rendering")
for c in ['full','k50']: clean(c)
