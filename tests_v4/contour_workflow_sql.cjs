// Run against an ephemeral PostgreSQL engine (PGlite), never production.
const { PGlite } = require(process.argv[2] || '@electric-sql/pglite');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
(async () => {
  const db = new PGlite();
  await db.exec(`create role anon; create role authenticated; create role service_role;
    create table users(id uuid primary key, role text);`);
  await db.exec(fs.readFileSync(path.join(__dirname,'../deploy-vps/migrations/001_contour_workflow.sql'),'utf8'));
  const owner='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', other='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
  const admin='dddddddd-dddd-4ddd-8ddd-dddddddddddd', aid='cccccccc-cccc-4ccc-8ccc-cccccccccccc';
  await db.query('insert into users values ($1,$2),($3,$4)',[owner,'user',admin,'admin']);
  const call = async (action,id,parent=null,actor=null,decision=null,reason='') =>
    (await db.query(`select contour_transition($1,$2,$3,$4,$5,$6,$7,$8,$9) as r`,
      [action,owner,id,aid,'image-hash',parent,actor,decision,reason])).rows[0].r;
  const first=await call('register','one');
  assert.deepEqual(await call('register','one'),first);
  await assert.rejects(call('register','competing-root'));
  assert.equal((await call('submit','one')).status,'submitted');
  await assert.rejects(call('decide','one',null,owner,'accepted'));
  await assert.rejects(call('decide','one',null,admin,'rejected',''));
  const accepted=await call('decide','one',null,admin,'accepted');
  assert.equal(accepted.status,'accepted');
  assert.deepEqual(await call('decide','one',null,admin,'accepted'),accepted);
  await assert.rejects(call('decide','one',null,admin,'rejected','different decision'));
  const child=await call('register','two','one');
  assert.equal(child.status,'draft');
  assert.equal(child.parent_id,'one');
  await assert.rejects(call('register','fork','one'));
  await db.query('select contour_transition($1,$2,$3,$4,$5)', ['register',other,'isolated',aid,'image-hash']);
  await assert.rejects(call('register','cross-owner','isolated'));
  await call('submit','two');
  const rejected=await call('decide','two',null,admin,'rejected','Contour misses crown');
  assert.equal(rejected.decisions.at(-1).reason,'Contour misses crown');
  assert.equal(rejected.decisions.at(-1).actor_id,admin);
  assert.ok(rejected.decisions.at(-1).at);
  assert.equal((await call('register','three','two')).status,'draft');
  assert.equal((await call('legacy','legacy-one')).status,'submitted');
  assert.equal((await call('legacy','legacy-two')).status,'submitted');
  // Privileges are part of the migration: no direct client-table or RPC access.
  const rights=await db.query(`select has_table_privilege('authenticated','contour_revisions','SELECT') as read,
    has_function_privilege('anon','contour_transition(text,uuid,text,uuid,text,text,uuid,text,text)','EXECUTE') as rpc`);
  assert.equal(rights.rows[0].read,false); assert.equal(rights.rows[0].rpc,false);
  // Rerunning the real migration preserves records, decisions and privileges.
  await db.exec(fs.readFileSync(path.join(__dirname,'../deploy-vps/migrations/001_contour_workflow.sql'),'utf8'));
  assert.deepEqual(await call('decide','one',null,admin,'accepted'),accepted);
  assert.equal((await db.query('select contour_workflow_version() as v')).rows[0].v,1);
  await db.exec('set role authenticated');
  await assert.rejects(db.query('select * from contour_revisions'));
  await assert.rejects(db.query('select contour_workflow_version()'));
  await assert.rejects(call('submit','two'));
  await db.exec('reset role');
  // Two queued writes cannot silently create competing children.
  const races=await Promise.allSettled([call('register','four-a','three'),call('register','four-b','three')]);
  assert.equal(races.filter(r=>r.status==='fulfilled').length,1);
  await db.close();
  console.log('PostgreSQL workflow checks passed: retries, conflicts, isolation, roles, decisions, legacy, race.');
})().catch(e=>{console.error(e);process.exitCode=1;});
