async page => {
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const assert=(ok,message)=>{if(!ok)throw Error(message)};
  await page.addInitScript(() => {
    const pixel='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aIu8AAAAASUVORK5CYII=';
    window.isTauri=true;
    window.__geoTest={calls:[],saved:null};
    const photo={id:'ui-contract-photo',source_name:'finished.jpg',width:4000,height:3000,preview:pixel,exif_location:{latitude:0,longitude:0}};
    window.__TAURI_INTERNALS__={invoke:async (command,args)=>{
      window.__geoTest.calls.push({command,args:JSON.parse(JSON.stringify(args??{}))});
      if(command==='get_api_port')throw Error('Sidecar deliberately unavailable in UI contract test');
      if(command==='plugin:dialog|open')return args.options.title.includes('项目')?'/tmp/ui-contract.imprint-geo.json':'/tmp/finished.jpg';
      if(command==='plugin:dialog|save')return args.options.title.includes('项目')?'/tmp/ui-contract.imprint-geo.json':'/tmp/ui-contract.png';
      if(command==='geo_open')return photo;
      if(command==='geo_preview')return {image:pixel,overlay:pixel,width:1600,height:1200,warning:'UI transport stub; no visual acceptance'};
      if(command==='geo_save_project'){window.__geoTest.saved=JSON.parse(JSON.stringify(args.config));return args.path;}
      if(command==='geo_load_project')return {photo:{...photo,id:'ui-contract-loaded'},config:window.__geoTest.saved};
      if(command==='geo_export')return args.path;
      throw Error('Unexpected command '+command);
    },transformCallback:()=>1,unregisterCallback:()=>{}};
  });
  await page.reload();
  await page.getByRole('button',{name:'地理叠加',exact:true}).click();
  const geo=page.getByTestId('geographic-overlay-page');
  await geo.getByRole('button',{name:'选择成片',exact:true}).first().click();
  await geo.getByText('finished.jpg',{exact:true}).waitFor();
  assert(await geo.getByText('来自照片 EXIF',{exact:true}).isVisible(),'zero-valued EXIF preserves GPS source');
  const enable=geo.getByRole('checkbox',{name:/启用地图叠加/});
  await enable.check();
  await page.waitForFunction(()=>window.__geoTest.calls.some(x=>x.command==='geo_preview'&&x.args.config.enabled));
  await geo.getByRole('img',{name:'finished.jpg 地理叠加效果',exact:true}).waitFor();
  await geo.getByRole('button',{name:'查看透明层',exact:true}).click();
  await geo.getByRole('img',{name:'原生返回的透明地理图层',exact:true}).waitFor();
  await geo.getByRole('button',{name:'保存项目…',exact:true}).click();
  await geo.getByText('项目已保存',{exact:true}).waitFor();
  const opacity=geo.locator('input[type=range]').first();
  await opacity.focus(); await opacity.press('ArrowRight'); await opacity.press('Tab');
  await geo.getByText('未保存',{exact:true}).waitFor();
  await geo.getByRole('button',{name:'撤销',exact:true}).click();
  await geo.getByText('项目已保存',{exact:true}).waitFor();
  await geo.getByRole('button',{name:'打开项目',exact:true}).click();
  await geo.getByText('项目已保存',{exact:true}).waitFor();
  await geo.locator('summary').filter({hasText:'导出'}).click();
  await geo.getByRole('button',{name:'导出全尺寸合成图',exact:true}).click();
  await page.waitForFunction(()=>window.__geoTest.calls.some(x=>x.command==='geo_export'&&!x.args.overlayOnly));
  await geo.getByRole('button',{name:'导出透明 PNG 图层',exact:true}).click();
  await page.waitForFunction(()=>window.__geoTest.calls.some(x=>x.command==='geo_export'&&x.args.overlayOnly));
  const calls=await page.evaluate(()=>window.__geoTest.calls.filter(x=>x.command.startsWith('geo_')));
  assert(calls.some(c=>c.command==='geo_preview'&&c.args.config.has_location&&c.args.config.latitude===0&&c.args.config.longitude===0),'real zero GPS reaches bridge');
  assert(calls.filter(c=>c.args.config).every(c=>'radius_km' in c.args.config&&typeof c.args.config.enabled==='boolean'),'snake_case config and booleans match Rust contract');
  assert(calls.filter(c=>c.command==='geo_export').every(c=>c.args.id==='ui-contract-loaded'),'exports use restored session');
  assert(errors.length===0,'page errors '+errors.join(';'));
  return {status:'passed',scope:'UI with explicit IPC test stub; does not prove native rendering',commands:calls.map(c=>c.command),page_errors:errors};
}
