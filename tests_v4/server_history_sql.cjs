const {PGlite}=require(process.argv[2] || '@electric-sql/pglite');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const db=new PGlite();
 await db.exec(`create role anon;create role authenticated;create role service_role;
 create table users(id uuid primary key,role text);
 create table analyses(id uuid primary key,user_id uuid,response_json jsonb);`);
 const migration=n=>fs.readFileSync(path.join(__dirname,'../deploy-vps/migrations/'+n),'utf8');
 await db.exec(migration('001_contour_workflow.sql'));
 await db.exec(migration('002_server_history.sql'));
 const a='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',b='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
 const id='cccccccc-cccc-4ccc-8ccc-cccccccccccc',v='11111111-1111-4111-8111-111111111111';
 const v2='22222222-2222-4222-8222-222222222222',v3='33333333-3333-4333-8333-333333333333';
 await db.query('insert into users values ($1,$2),($3,$2)',[a,'user',b]);
 const save=async(owner,version,parent=null,hash='a'.repeat(64),image='b'.repeat(64),correction=null)=>
 (await db.query('select save_report_version($1,$2,$3,$4,$5,$6,$7,$8) r',[owner,id,version,parent,hash,image,correction,{}])).rows[0].r;
 const first=await save(a,v);
 assert.deepEqual(await save(a,v),first);
 await assert.rejects(save(a,v,null,'c'.repeat(64)));
 await assert.rejects(save(a,v2)); // second root is a conflict
 await assert.rejects(save(b,v2,v)); // cross-owner parent
 await assert.rejects(save(a,v2,v,'c'.repeat(64),'d'.repeat(64))); // changed original
 await save(a,v2,v);
 await assert.rejects(save(a,v3,v)); // two devices based on same parent
 await assert.rejects(save(a,v3,v2,'d'.repeat(64),'b'.repeat(64),'unknown-contour'));
 await db.query('select contour_transition($1,$2,$3,$4,$5)',['register',a,'contour',id,'b'.repeat(64)]);
 await save(a,v3,v2,'d'.repeat(64),'b'.repeat(64),'contour');
 assert.equal((await db.query('select status from contour_revisions')).rows[0].status,'draft');
 await db.exec(migration('002_server_history.sql'));
 assert.equal((await db.query('select count(*)::int n from report_versions')).rows[0].n,3);
 await db.exec('set role authenticated');
 await assert.rejects(db.query('select * from report_versions'));
 await assert.rejects(db.query('select server_history_version()'));
 await db.exec('reset role');
 const grants=await db.query(`select has_function_privilege('authenticated',
 'save_report_version(uuid,uuid,uuid,uuid,text,text,text,jsonb)','EXECUTE') e`);
 assert.equal(grants.rows[0].e,false);
 assert.equal((await db.query('select server_history_version() n')).rows[0].n,1);
 await db.close(); console.log('PASS: real PostgreSQL engine, retries, conflicts, owner isolation, ACL, rerun, contour unchanged');
})().catch(e=>{console.error(e);process.exitCode=1});
