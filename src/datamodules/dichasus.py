from pathlib import Path

import lightning as l
import requests
from tqdm import tqdm


class DICHASUSDataModule(l.LightningDataModule):
    ALL_FILES = {
        'dichasus-cf02.tfrecords': 'doi:10.18419/darus-2854/8',
        'dichasus-cf03.tfrecords': 'doi:10.18419/darus-2854/9',
        'dichasus-cf04.tfrecords': 'doi:10.18419/darus-2854/10',
        'dichasus-cf05.tfrecords': 'doi:10.18419/darus-2854/11',
        'dichasus-cf06.tfrecords': 'doi:10.18419/darus-2854/12',
        'dichasus-cf07.tfrecords': 'doi:10.18419/darus-2854/13',
    }

    def __init__(self, files=None, data_dir='dichasus'):
        super().__init__()
        self.files = files or list(self.ALL_FILES.keys())
        self.data_dir = Path(data_dir)

    def prepare_data(self):
        base_url = 'https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId='

        for filename in self.files:
            doi = self.ALL_FILES[filename]
            dest = self.data_dir / filename
            if dest.exists():
                print(f'Already exists: {filename}')
                continue

            r = requests.get(base_url + doi, stream=True)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            dest.parent.mkdir(parents=True, exist_ok=True)
            with (
                open(dest, 'wb') as f,
                tqdm(total=total, unit='B', unit_scale=True, desc=filename) as bar,
            ):
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
