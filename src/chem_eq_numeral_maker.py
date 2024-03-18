import os
import io
import time
import random

from datetime import datetime
from PIL import Image, ImageOps
from matplotlib import rcParams

import matplotlib.pyplot as plt


class ChemEqMaker(object):
    def __init__(self, max_n_compounds, max_n_elements, max_n_quantity):
        self.min_n_compounds = 1
        self.max_n_compounds = max_n_compounds
        self.min_n_elements = 1
        self.max_n_elements = max_n_elements
        self.min_n_quantity = 1
        self.max_n_quantity = max_n_quantity

        self.elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
            'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
            'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
            'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
            'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
            'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
        self.joiners = ['+', 'plus', 'and', 'with']

    def make_record(self):
        return self.make_equation()

    def make_equation(
            self,
            # min_n_compounds=1, max_n_compounds=4, min_elements_per_compound=1,
            # max_elements_per_compound=4, subscript_min=1, subscript_max=500
    ):
        eq_components = []
        gt_eq_components = []
        n_compounds = random.randint(self.min_n_compounds, self.max_n_compounds + 1)

        for compound_no in range(n_compounds):
            compound, gt_compound = self.make_compound(
                self.min_n_elements, self.max_n_elements, self.min_n_quantity, self.max_n_quantity
            )
            eq_components.append(compound)
            gt_eq_components.append(gt_compound)

            if compound_no < n_compounds - 1:
                joiner = random.choice(self.joiners)
                eq_components.append(joiner)
                joiner_gt_str = []
                for char in joiner:
                    joiner_gt_str.append(char)

                joiner_gt_str = ' '.join(joiner_gt_str)
                gt_eq_components.append(joiner_gt_str)

        eq = ' '.join(eq_components)
        gt_eq = ' \; '.join(gt_eq_components)
        return eq, gt_eq

    def make_compound(self, n_elements_min, n_elements_max, sub_min, sub_max):
        components = []
        gt_components = []
        n_elements = random.randint(n_elements_min, n_elements_max + 1)
        # element_usage = {}
        used_elements = [None]
        for _ in range(n_elements):
            element = None
            while element in used_elements:
                element = random.choice(self.elements)
                # unused_element = element_usage.get(element, False)

            used_elements.append(element)
            components.append(element)
            components.append('_{')
            for char in element:
                gt_components.append(char)
            gt_components.append('_')
            gt_components.append('{')

            subscript = random.randint(sub_min, sub_max)
            subscript = str(subscript)
            components.append(subscript)
            for char in subscript:
                gt_components.append(char)

            components.append('}')
            gt_components.append('}')

        compound = '${}$'.format(''.join(components))
        gt_compound = ' '.join(gt_components)
        return compound, gt_compound


class NumeralMaker(object):
    def __init__(self, decimal_p=0.5, max_numbers=4, symbol_p=0.1):
        self.decimal_p = decimal_p
        self.max_numbers = max_numbers
        self.symbol_p = symbol_p

        self.common_eq_symbols = [
            '\\mu', '\\alpha', '\\partial', '\\nu', '\\pi', '\\phi', '\\delta', '\\lambda', '\\beta',
            '\\theta', '\\sigma', '\\gamma', '\\psi', '\\sigma', '\\tau', '\\sum', '\\omega', '\\epsilon', '\\eta',
            '\\xi', '\\Phi', '\\Gamma', '\\infty', '\\Lambda', '\\varphi', '\\Delta', '\\chi', '\\Omega',
            '\\kappa', '\\Psi', '\\zeta', '\\varepsilon',  '\\Pi', '\\Theta',
        ]

    def make_float(self):
        num = random.randint(0, 100000)

        if random.random() <= self.decimal_p:
            dec = []
            n_dec = random.randint(1, 4)
            for _ in range(n_dec):
                dec.append(random.randint(0, 9))
            dec = ''.join([str(v) for v in dec])

            num = float('{}.{}'.format(num, dec))

        return num

    def make_record(self):
        return self.make_numeral()

    def make_numeral(self):
        text = []
        label = []

        n_components = random.randint(1, self.max_numbers)
        n_joiners = n_components - 1
        components = []
        joiners = []
        joiner_options = ['\pm', '+', '-', '/', '\quad', '\\neq']

        for _ in range(n_components):
            if random.random() <= (1 - self.symbol_p):
                components.append(self.make_float())
            else:
                components.append(random.choice(self.common_eq_symbols))

        if random.random() <= 0.01:
            joiners = ['\quad' for _ in range(n_joiners)]
        else:
            for _ in range(n_joiners):
                joiners.append(random.choice(joiner_options))

        for idx in range(n_components):
            this_component = components[idx]
            if type(this_component) in [float, int]:
                this_component = '{:,}'.format(this_component)
            text.append(this_component)
            if this_component.startswith('\\'):
                label.append(this_component)
            else:
                label.extend([t for t in this_component])

            if idx < n_components - 1:
                this_joiner = joiners[idx]
                text.append(this_joiner)
                label.append(this_joiner)

        text = ' '.join([str(v) for v in text])
        label = ' '.join([str(v) for v in label])

        text = r'${}$'.format(text)

        return text, label


class ChemEqNumeralMaker(object):
    def __init__(
            self, data_soure, outdir, record_idx_start=0, out_q=None, worker_idx=0, available_font_names=None,
            fontsizes=None, fontsize_weights=None, max_w=600, numeral_decimal_p=0.5, numeral_max_numerals=4,
            numeral_symbol_p=0.1, chemeq_n_compound=4, chemeq_n_elements=4, chemeq_n_quantity=500
    ):
        self.outdir = outdir
        self.record_idx_start = record_idx_start
        self.out_q = out_q
        self.worker_idx = worker_idx
        self.available_font_names = available_font_names
        self.fontsizes = fontsizes
        self.fontsize_weights = fontsize_weights
        self.max_w = max_w

        self.base_filename_tmplt = 'render_{}.jpg'
        self.final_out = os.path.join(self.outdir, 'final_renders')

        self.data_source = data_soure
        if data_soure == 'chemeq':
            print('RecordMaker {} creating ChemEq generator'.format(self.worker_idx))
            self.generator = ChemEqMaker(
                chemeq_n_compound, chemeq_n_elements, chemeq_n_quantity
            )
        else:
            print('RecordMaker {} creating Numeral generator'.format(self.worker_idx))
            self.generator = NumeralMaker(
                numeral_decimal_p, numeral_max_numerals, numeral_symbol_p
            )

    def generate_many_record_and_process(self, n):
        label_d = {}
        font_d = {}
        n_processed = 0
        summary_every = max(n // 20, 1)
        rcParams['text.usetex'] = False
        start_time = time.time()

        while n_processed < n:
            try:
                record_text, record_label = self.generator.make_record()
                selected_font = random.choice(self.available_font_names)
                selected_fontsize = random.choices(self.fontsizes, self.fontsize_weights, k=1)[0]
                valid_size = False
                base_filename = self.base_filename_tmplt.format(self.record_idx_start + n_processed)

                while not valid_size:
                    plt.text(0.0, 0.5, record_text, fontsize=selected_fontsize, fontname=selected_font)
                    plt.axis('off')
                    plt.box(False)

                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='jpg', bbox_inches='tight')
                    plt.clf()

                    image = Image.open(img_buf)
                    image.load()
                    inverted_image = ImageOps.invert(image.convert("RGB"))
                    cropped = image.crop(inverted_image.getbbox())

                    if cropped.size[0] > self.max_w:
                        # print('Image too big! Lowering font size')
                        selected_fontsize -= 1
                    else:
                        valid_size = True

                out_fp = os.path.join(self.final_out, base_filename)
                cropped.save(out_fp)

                label_d[base_filename] = record_label
                font_d[base_filename] = {
                    'font': selected_font,
                    'fontsize': selected_fontsize,
                    'source': 'numeral',
                }

                # curr_idx += 1
                n_processed += 1
                if n_processed % summary_every == 0:
                    elapsed_time = time.time() - start_time
                    print('[{0}] Worker {1} created {2} of {3} {4} records... avg {5:.4}s/record'.format(
                        datetime.now().strftime("%H:%M:%S"), self.worker_idx,
                        n_processed, n, self.data_source, elapsed_time / n_processed
                    ))

            except RuntimeError as ex:
                print('** [{0}] Worker {1} error processing line \'{2}\' w/ gt \'{3}\' from {4} **'.format(
                    datetime.now().strftime("%H:%M:%S"), self.worker_idx, record_text, record_label, self.data_source
                ))

        elapsed_time = time.time() - start_time
        print('[{0}] Worker {1} made {2} records...  avg {3:.4}s/record... pushing results to out_q'.format(
            datetime.now().strftime("%H:%M:%S"), self.worker_idx, n_processed, elapsed_time / n_processed)
        )
        self.out_q.put([self.worker_idx, label_d, font_d])

