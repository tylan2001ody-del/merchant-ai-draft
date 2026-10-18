"""Record a real CLI run as an asciinema v2 terminal recording (stdlib only)."""
import codecs
import json
import os
from pathlib import Path
import pty
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
command = [sys.executable, str(root/'mn_draft.py'), '--input', 'examples/before.json']
master, slave = pty.openpty()
started = time.monotonic()
recording = root/'docs/demo.cast'
with recording.open('w', encoding='utf-8') as f:
    f.write(json.dumps({'version':2,'width':110,'height':30,
                        'title':'Offline Chinese to Cyrillic Mongolian merchant draft',
                        'env':{'TERM':'xterm-256color'}})+'\n')
    def event(text):
        f.write(json.dumps([round(time.monotonic()-started,3),'o',text],ensure_ascii=False)+'\n')
        f.flush()
        print(text,end='',flush=True)
    event('$ python mn_draft.py --input examples/before.json\r\n')
    child = subprocess.Popen(command, cwd=root, stdout=slave, stderr=slave)
    os.close(slave)
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    try:
        while True:
            try:
                chunk = os.read(master, 8192)
            except OSError:
                break
            if not chunk:
                break
            event(decoder.decode(chunk))
        event(decoder.decode(b'',final=True))
    finally:
        os.close(master)
    code = child.wait()
    event(f'\r\nExit status: {code}\r\n')
    if code == 0:
        shutil.copyfile(root/'examples/before.json.mn-draft.json', root/'examples/after.json')
        event('$ cat examples/after.json\r\n')
        event((root/'examples/after.json').read_text().replace('\n','\r\n'))
sys.exit(code)
