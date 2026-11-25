(function(root){
 'use strict';
 const publicFields=['item_id','image','capability','objects','statement_a','statement_b'];
 const ratingFields=['objects_present','observed_count','relation_a','relation_b','object_form','notes'];
 const fields=[...publicFields,...ratingFields],choices=['yes','no','unclear'];
 const blank=items=>items.map(x=>({...x,...Object.fromEntries(ratingFields.map(f=>[f,'']))}));
 function error(r){
  if(!choices.includes(r.objects_present)||!['coherent','deformed','unclear'].includes(r.object_form))return 'Record whether trucks are present and whether their shapes are intact.';
  if(r.capability==='counting'){
   if(r.observed_count==='unclear'){if(r.objects_present==='no')return 'Enter 0 if there are no trucks.';}
   else if(r.objects_present==='unclear'){if(r.observed_count!=='')return 'Leave the count blank or select Unsure if you cannot identify the trucks.';}
   else if(!/^(0|[1-9][0-9]?)$/.test(r.observed_count))return 'Enter a whole number from 0 to 99.';
   else if((r.objects_present==='no')!==(Number(r.observed_count)===0))return 'Check that the count agrees with your answer about truck presence.';
   if(r.relation_a||r.relation_b)return 'Counting tasks do not include relations.';
  }else{
   if(r.observed_count)return 'Spatial tasks do not include a count.';
   if(!choices.includes(r.relation_a)||!choices.includes(r.relation_b))return 'Evaluate both spatial statements.';
   if(r.objects_present==='no'&&(r.relation_a!=='no'||r.relation_b!=='no'))return 'Mark both relations as absent when the target objects are missing.';
   if(r.objects_present==='unclear'&&(r.relation_a==='yes'||r.relation_b==='yes'))return 'Mark a relation as absent or unclear when its objects are unclear.';
  }
  if([r.objects_present,r.observed_count,r.object_form,r.relation_a,r.relation_b].includes('unclear')&&!r.notes.trim())return 'Add a brief note when a judgment is unclear.';
  return '';
 }
 const complete=rows=>rows.filter(r=>!error(r)).length;
 function csv(rows){
  if(!rows.length||complete(rows)!==rows.length)throw Error('incomplete ratings');
  const q=x=>/[",\r\n]/.test(x)?'"'+x.replaceAll('"','""')+'"':x;
  return '\uFEFF'+fields.join(',')+'\r\n'+rows.map(r=>fields.map(f=>q(r[f])).join(',')).join('\r\n')+'\r\n';
 }
 function restore(packet,d){
  if(d.packet!==packet.id||!Array.isArray(d.rows)||d.rows.length!==packet.items.length||
     !Number.isInteger(d.index)||d.index<0||d.index>=packet.items.length)throw Error('This draft belongs to a different evaluation.');
  d.rows.forEach((r,i)=>{
   if(Object.keys(r).sort().join('|')!==[...fields].sort().join('|')||publicFields.some(f=>r[f]!==packet.items[i][f])||
      ratingFields.some(f=>typeof r[f]!=='string'))throw Error('The draft does not match the images and questions in this evaluation.');
   for(const f of ['objects_present','relation_a','relation_b'])if(!['',...choices].includes(r[f]))throw Error('The draft contains an invalid judgment.');
   if(!['','coherent','deformed','unclear'].includes(r.object_form))throw Error('The draft contains an invalid shape judgment.');
  });
  return d;
 }
 const api={publicFields,ratingFields,fields,blank,error,complete,csv,restore};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.DonorAudit=api;
})(typeof globalThis!=='undefined'?globalThis:this);
