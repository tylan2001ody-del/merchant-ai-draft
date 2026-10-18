import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
import mn_draft as cli

ROOT = Path(__file__).resolve().parents[1]

class DraftTests(unittest.TestCase):
    def setUp(self):
        self.before = json.loads((ROOT/'examples/before.json').read_text())

    def run_cli(self, data, factory, in_place=False):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)/'merchant.json'
        path.write_text(json.dumps(data, ensure_ascii=False))
        original = path.read_bytes()
        error = io.StringIO()
        with redirect_stderr(error):
            rc = cli.main(['--input', str(path)] + (['--in-place'] if in_place else []), factory)
        output = path if in_place else Path(str(path)+'.mn-draft.json')
        return rc, output, error.getvalue(), path, original

    def test_fills_only_allowlist_preserves_input_and_references(self):
        seen = []
        def translate(text):
            seen.append(text)
            return 'Монгол '+text
        rc, target, log, path, original = self.run_cli(self.before, lambda: translate)
        self.assertEqual(rc, 0)
        self.assertEqual(path.read_bytes(), original)
        out = json.loads(target.read_text())
        self.assertEqual(out['translationStatus']['mn'], 'draft')
        self.assertEqual(len(seen), 9)
        zh, mn = out['texts']['zh'], out['texts']['mn']
        self.assertEqual(mn['mapUrl'], zh['mapUrl'])
        self.assertEqual(mn['products'][0]['imageId'], zh['products'][0]['imageId'])
        for key in ('products','services'):
            self.assertEqual([x['id'] for x in mn[key]], [x['id'] for x in zh[key]])
        self.assertNotIn(zh['mapUrl'], seen)
        self.assertNotIn('demo-noodles', seen)
        self.assertNotIn('demo-bowl', seen)
        for key in self.before:
            if key not in ('texts','translationStatus'):
                self.assertEqual(out[key],self.before[key])

    def test_reviewed_skips_loading_and_keeps_bytes(self):
        data = copy.deepcopy(self.before)
        data['translationStatus']['mn'] = 'reviewed'
        data['texts']['mn'] = {'name':'Баталгаажсан'}
        def forbidden():
            raise AssertionError('model must not load')
        for in_place in (False,True):
            rc, target, _, path, original = self.run_cli(data,forbidden,in_place)
            self.assertEqual(rc,0)
            self.assertEqual(path.read_bytes(),original)
            if not in_place:
                self.assertFalse(target.exists())

    def test_existing_mongolian_never_overwritten_and_array_matches_id(self):
        data = copy.deepcopy(self.before)
        data['texts']['zh']['products'].append({'id':'second','name':'第二个商品'})
        data['texts']['mn'] = {'name':'原有名称', 'products':[
            {'id':'second','name':'Хоёр'}, {'id':'demo-noodles','name':'Нэг'}]}
        out,jobs = cli.prepare(data)
        cli.apply_jobs(out,jobs,lambda s:'译文',lambda _:None)
        self.assertEqual(out['texts']['mn']['name'],'原有名称')
        self.assertEqual([x['name'] for x in out['texts']['mn']['products']],['Нэг','Хоёр'])
        self.assertIsNone(data['texts']['mn']['products'][1].get('imageId'))

    def test_failure_retry_preserves_whitespace_and_continues(self):
        data = copy.deepcopy(self.before)
        data['texts']['mn'] = {'summary':'  ', 'name':'已有'}
        attempts=[]
        def translate(text):
            attempts.append(text)
            if text == data['texts']['zh']['summary']:
                raise RuntimeError('injected failure')
            return 'Монгол'
        rc,target,log,_,_ = self.run_cli(data,lambda:translate)
        self.assertEqual(rc,1)
        out=json.loads(target.read_text())
        self.assertEqual(out['texts']['mn']['summary'],'  ')
        self.assertEqual(out['texts']['mn']['name'],'已有')
        self.assertEqual(out['texts']['mn']['location'],'Монгол')
        self.assertEqual(attempts.count(data['texts']['zh']['summary']),2)
        self.assertIn('texts.mn.summary',log)

    def test_missing_failure_stays_missing(self):
        out,jobs=cli.prepare(self.before)
        def fail(_):
            raise RuntimeError('no translation')
        self.assertEqual(cli.apply_jobs(out,jobs,fail,lambda _:None),9)
        self.assertNotIn('name',out['texts']['mn'])
        self.assertNotIn('name',out['texts']['mn']['products'][0])

    def test_load_failure_no_output_no_input_change(self):
        def fail():
            raise RuntimeError('empty offline cache')
        rc,target,log,path,original=self.run_cli(self.before,fail,True)
        self.assertEqual(rc,2)
        self.assertIn('model load',log)
        self.assertEqual(path.read_bytes(),original)

    def test_reference_conflicts_and_extra_ids_are_rejected(self):
        for mn in ({'mapUrl':'https://example.com/other'},
                   {'products':[{'id':'unknown','name':'Keep me'}]},
                   {'products':[{'id':'demo-noodles','imageId':'other'}]},
                   {'services':[{'id':'x'},{'id':'x'}]}):
            data=copy.deepcopy(self.before)
            data['texts']['mn']=mn
            with self.assertRaises(cli.DraftError):cli.prepare(data)

    def test_unknown_fields_preserved_not_translated(self):
        data=copy.deepcopy(self.before)
        data['texts']['zh']['futureField']='不要翻译未知字段'
        out,jobs=cli.prepare(data)
        self.assertNotIn('不要翻译未知字段',[j[2] for j in jobs])
        self.assertEqual(out['texts']['zh']['futureField'],'不要翻译未知字段')

    def test_long_input_no_loss_and_unsplittable_fails(self):
        text='甲'*300+'。'+'乙'*300+'！'+'丙'*300+'？'
        chunks=cli.split_input(text,len)
        self.assertEqual(''.join(chunks),text)
        self.assertTrue(all(len(x)<=512 for x in chunks))
        with self.assertRaises(cli.DraftError): cli.split_input('甲'*600,len)

    def test_atomic_replace_failure_leaves_input_and_no_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'merchant.json'
            path.write_text('original')
            with patch('mn_draft.os.replace',side_effect=OSError('injected')):
                with self.assertRaises(OSError):cli.atomic_write(path,{'a':1})
            self.assertEqual(path.read_text(),'original')
            self.assertEqual(list(Path(tmp).iterdir()),[path])

    def test_bad_status_returns_clear_error(self):
        data=copy.deepcopy(self.before)
        data['translationStatus']=None
        rc,target,log,_,_=self.run_cli(data,lambda:None)
        self.assertEqual(rc,2)
        self.assertFalse(target.exists())
        self.assertIn('translationStatus',log)

if __name__=='__main__':unittest.main()
