const fs = require('fs');
const vm = require('vm');
const sourcePath = require('path').join(__dirname, '../foundry/review/web/app.js');
const assert = require('node:assert/strict');
const raw = fs.readFileSync(sourcePath, 'utf8');
const source = raw.replace(/\}\)\(\);\s*$/, 'globalThis.audit = {state,dom,saveBbox,saveQueryEdit,goToIndex,isAiAnnotator,isTodoItem,markTodo};\n})();');
if (source === raw) throw new Error('Instrumentation did not match the source terminator');

function harness() {
  const calls = [];
  const nodes = new Map();
  const canvas = new Proxy({measureText:()=>({width:10})}, {
    get(target,key) { return key in target ? target[key] : (()=>{}); },
    set(target,key,value) { target[key]=value; return true; }
  });
  function node(id) {
    if (nodes.has(id)) return nodes.get(id);
    const n = {id, tagName:id==='query-en-text'?'INPUT':'DIV', value:'', dataset:{}, style:{},
      width:800,height:600,classList:{add(){},remove(){}},
      getContext:()=>canvas,getBoundingClientRect:()=>({width:800,height:600,left:0,top:0}),
      addEventListener(){},appendChild(){},remove(){},focus(){},blur(){},setSelectionRange(){}};
    nodes.set(id,n); return n;
  }
  const context={
    console, URL, Image:class {}, setTimeout:()=>0,
    localStorage:{getItem:()=>'',setItem(){}},history:{replaceState(){}},
    window:{location:{hash:'',origin:'http://localhost'},devicePixelRatio:1,addEventListener(){}},
    document:{readyState:'loading',getElementById:node,createElement:()=>node('toast'),addEventListener(){}},
    fetch:(url,options)=>new Promise(resolve=>calls.push({url,options,resolve}))
  };
  vm.createContext(context);
  vm.runInContext(source,context,{filename:sourcePath});
  const a=context.audit;
  a.state.items=[
    {id:'a',bbox:[0.1,0.1,0.3,0.3],annotator:'glm-4.6v',image_url:'/a',query_en:'Query A',frame_id:'a',ordinal:1},
    {id:'b',bbox:[0.6,0.6,0.9,0.9],annotator:'glm-4.6v',image_url:'/b',query_en:'Query B',frame_id:'b',ordinal:1}
  ];
  a.state.reviewMode=true;
  a.state.annotator='reviewer'; a.dom.annotatorInput.value='reviewer';
  for (const item of a.state.items) a.state.imageCache.set(item.image_url,{naturalWidth:100,naturalHeight:100});
  a.goToIndex(0);
  return {a,calls};
}

async function main() {
  {
    const {a,calls}=harness();
    const pending=a.saveBbox();
    const sent=JSON.parse(calls[0].options.body).bbox;
    a.goToIndex(1);
    calls[0].resolve({ok:true,json:async()=>({id:'a',bbox:sent})});
    await pending;
    assert.equal(a.state.currentIndex,1);
    assert.equal(JSON.stringify(a.state.activeBbox),JSON.stringify(a.state.items[1].bbox));
  }
  {
    const {a,calls}=harness();
    const pending=a.saveBbox();
    const sent=JSON.parse(calls[0].options.body).bbox;
    a.goToIndex(1); a.goToIndex(0);
    const draft=[0.2,0.3,0.4,0.5]; a.state.activeBbox=draft;
    calls[0].resolve({ok:true,json:async()=>({id:'a',bbox:sent})});
    await pending;
    assert.equal(a.state.currentIndex,0);
    assert.equal(JSON.stringify(a.state.activeBbox),JSON.stringify(draft));
  }
  {
    const {a,calls}=harness();
    a.dom.queryEnText.value='Edited A';
    const pending=a.saveQueryEdit();
    a.goToIndex(1);
    calls[0].resolve({ok:true,json:async()=>({})});
    await pending;
    assert.equal(a.dom.queryEnText.dataset.original,'Query B');
    assert.equal(a.dom.queryEnText.value,'Query B');
    assert.equal(a.state.items[0].query_en,'Edited A');
  }
  {
    const {a,calls}=harness();
    a.dom.queryEnText.value='Edited A';
    const first=a.saveQueryEdit();
    const blur=a.saveQueryEdit();
    assert.equal(calls.length,1);
    calls[0].resolve({ok:true,json:async()=>({})});
    await Promise.all([first,blur]);
  }
  {
    const {a,calls}=harness();
    a.dom.queryEnText.value='First edit';
    const first=a.saveQueryEdit();
    a.dom.queryEnText.value='Query A'; // revert while the first save is pending
    const second=a.saveQueryEdit();
    assert.equal(calls.length,1);
    calls[0].resolve({ok:true,json:async()=>({})});
    await first;
    await Promise.resolve();
    assert.equal(calls.length,2);
    assert.equal(JSON.parse(calls[1].options.body).query,'Query A');
    calls[1].resolve({ok:true,json:async()=>({})});
    await second;
    assert.equal(a.state.items[0].query_en,'Query A');
    assert.equal(a.dom.queryEnText.dataset.original,'Query A');
  }
  {
    const {a,calls}=harness();
    const pending=a.markTodo();
    a.goToIndex(1);
    calls[0].resolve({ok:true,json:async()=>({})});
    await pending;
    assert.equal(a.state.currentIndex,1);
  }
  {
    const {a}=harness();
    for(const name of ['reviewer','Eric','Kai','Bai']) {
      assert.equal(a.isAiAnnotator(name),false);
      assert.equal(a.isTodoItem({bbox:[0,0,1,1],annotator:name}),false);
    }
    assert.equal(a.isAiAnnotator('glm-4.6v'),true);
  }
  console.log('frontend state tests passed');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
