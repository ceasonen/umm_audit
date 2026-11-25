const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs');
function core(){assert.ok(fs.existsSync('site/audit/audit-core.js'),'donor web core missing');return require('../site/audit/audit-core.js');}
test('fresh human packet stays blank; incomplete export fails',()=>{
 const c=core(),rows=c.blank([{item_id:'x',image:'images/x.png',capability:'counting',objects:'cats',statement_a:'',statement_b:''}]);
 assert.equal(c.complete(rows),0);assert.throws(()=>c.csv(rows),/incomplete/);
 rows[0].objects_present='no';rows[0].observed_count='0';rows[0].object_form='coherent';
 assert.equal(c.complete(rows),1);assert.ok(c.csv(rows).includes('observed_count'));
 rows[0].observed_count='3';assert.equal(c.complete(rows),0);
});
test('draft restore rejects another packet or altered public identity',()=>{
 const c=core(),p={id:'rater1',items:[{item_id:'x',image:'x.png',capability:'spatial',objects:'cup; plate',statement_a:'left',statement_b:'right'}]};
 const d={packet:p.id,rows:c.blank(p.items),index:0};assert.deepEqual(c.restore(p,d),d);
 assert.throws(()=>c.restore(p,{...d,packet:'rater2'}));
 assert.throws(()=>c.restore(p,{...d,rows:[{...d.rows[0],objects:'changed'}]}));
});
test('count uncertainty is separate from object presence',()=>{
 const c=core();assert.equal(c.error({capability:'counting',objects_present:'yes',observed_count:'unclear',relation_a:'',relation_b:'',object_form:'deformed',notes:'occlusion'}),'');
});
