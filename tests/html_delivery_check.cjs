// Execute only the offline document's navigation/search functions against a
// small deterministic DOM fixture. This is a source test, not a browser preview.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const [file, report] = process.argv.slice(2);
const source = fs.readFileSync(file, 'utf8');
const decode = s => s.replace(/<[^>]*>/g,' ').replace(/&#x27;/g,"'").replace(/&quot;/g,'"').replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>');
assert(!/<script[^>]+src=|<link[^>]+href=['"]https?:|<img[^>]+src=['"]https?:/i.test(source));
const sections = [...source.matchAll(/<section id='([^']+)' class='([^']*)'>([\s\S]*?)<\/section>/g)].map(m => {
  const values = new Set(m[2].split(' '));
  return {id:m[1], textContent:decode(m[3]), classList:{contains:x=>values.has(x),remove:x=>values.delete(x),add:x=>values.add(x)}};
});
assert.equal(new Set(sections.map(x=>x.id)).size,sections.length);
const links = [...source.matchAll(/<a\b([^>]*class='nav-item'[^>]*)>([\s\S]*?)<\/a>/g)].map(m=> {
  const href=m[1].match(/href='([^']+)'/)[1];
  return {textContent:decode(m[2]),hidden:false,getAttribute:name=>name==='href'?href:null};
});
assert(links.length>=60);
const index = Object.fromEntries(sections.map(x=>[x.id,x]));
for(const link of links)assert(index[link.getAttribute('href').slice(1)]);
index['search-count']={textContent:''}; index['api-search']={value:''};
const events={};
const context=vm.createContext({document:{getElementById:x=>index[x],querySelectorAll:x=>x==='.doc-section'?sections:links},location:{hash:''},addEventListener:(key,fn)=>events[key]=fn});
const scripts=[...source.matchAll(/<script>([\s\S]*?)<\/script>/g)];assert.equal(scripts.length,1);
vm.runInContext(scripts[0][1],context,{timeout:1000});
events.load();assert.equal(links.filter(x=>!x.hidden).length,links.length);
for(const id of ['yolo_compatibility','YOLO_LoadModelFromPath','YOLO_InferJson','yolo_delivery']) {
  context.show(id);assert(index[id].classList.contains('active'));
  assert.equal(sections.filter(x=>x.classList.contains('active')).length,1);
  assert.equal(context.location.hash,id);
}
context.filterNav('yolo26');assert(links.some(x=>!x.hidden && x.getAttribute('href')==='#yolo_compatibility'));
context.filterNav('nonexistent-test-query-abcdef');assert(links.every(x=>x.hidden));
assert.equal(index['search-count'].textContent,'找到 0 项');
context.filterNav('');assert(links.every(x=>!x.hidden));
context.location.hash='#YOLO_InferJson';events.load();assert(index.YOLO_InferJson.classList.contains('active'));
const result={passed:true,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),sections:sections.length,navigation_links:links.length,offline_assets:true,navigation:true,search:true,hash_navigation:true,method:'actual inline JavaScript with deterministic DOM fixture',visual_browser_preview:'not performed by this deterministic DOM test'};
if(report)fs.writeFileSync(report,JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
