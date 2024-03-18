import os
import io
import re
import time
import random

import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib import rcParams

from PIL import Image, ImageOps

from .utils import no_arg_latex_symbols, one_arg_latex_symbols, find_bad_brackets, convert_str_to_label_format


class PrintedEnglishMaker(object):
    def __init__(self, args, arxiv_data, pubmed_data, chemrxiv_data, outdir, record_idx_start=0,
                 superscript_p=0.0375, subscript_p=0.0125, newline_p=0.2, max_newlines=4, out_q=None, worker_idx=0,
                 available_font_names=None, fontsizes=None, fontsize_weights=None, latex_insertion_p=0.15,
                 max_w=600, in_q=None
                 ):
        self.available_font_names = available_font_names
        self.fontsizes = fontsizes
        self.fontsize_weights = fontsize_weights
        self.max_w = max_w
        self.in_q = in_q

        self.superscript_p = superscript_p
        self.subscript_p = subscript_p
        self.newline_p = newline_p
        self.latex_insertion_p = latex_insertion_p
        self.out_q = out_q
        self.worker_idx = worker_idx

        self.intermediate_out = os.path.join(outdir, 'intermediate_renders')
        self.final_out = os.path.join(args.out, 'final_renders')

        self.record_idx_start = record_idx_start

        self.arxiv_documents = []
        self.pubmed_documents = []
        self.chemrxiv_documents = []

        self.n_newline_options = [i for i in range(1, max_newlines)]
        self.n_newline_weights = [1 / (i * i) for i in range(1, max_newlines)]
        self.arxiv_documents = arxiv_data
        print('PrintedEnglishMaker {} recvd {} arXiv documents'.format(self.worker_idx, len(self.arxiv_documents)))

        self.pubmed_documents = pubmed_data
        print('PrintedEnglishMaker {} recvd {} PubMed documents'.format(self.worker_idx, len(self.pubmed_documents)))

        self.chemrxiv_documents = chemrxiv_data
        print('PrintedEnglishMaker {} recvd {} Chemrxiv documents'.format(self.worker_idx, len(self.chemrxiv_documents)))

        self.base_filename_tmplt = 'render_{}.jpg'
        self.common_sup_sub_symbols = [
            '\\mu', '\\alpha', '\\nu', '\\pi', '\\phi', '\\delta', '\\lambda', '\\beta',
            '\\theta', '\\sigma', '\\gamma', '\\psi', '\\sigma', '\\tau', '\\sum', '\\omega', '\\epsilon', '\\eta',
            '\\xi', '\\Phi', '\\Gamma', '\\infty', '\\Lambda', '\\varphi', '\\Delta', '\\chi', '\\Omega',
            '\\kappa', '\\Psi', '\\zeta', '\\varepsilon', '\\Pi', '\\Theta',
        ]
        self.letters = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'w', 'x', 'y', 'z'
        ]
        self.alphanumeric = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '[', ']', '(', ')', '\\mu', '\\alpha', '\\nu', '\\pi', '\\phi', '\\delta', '\\lambda', '\\beta',
            '\\theta', '\\sigma', '\\gamma', '\\psi', '\\sigma', '\\tau', '\\omega', '\\epsilon', '\\eta',
            '\\xi', '\\Phi', '\\Gamma', '\\infty', '\\Lambda', '\\varphi', '\\Delta', '\\chi', '\\Omega',
            '\\kappa', '\\Psi', '\\zeta', '\\varepsilon', '\\Pi', '\\Theta',
        ]

    def generate_and_process_from_q(self, words_min=1, words_max=12, summary_every=1000):
        n_processed = 0
        all_done = False
        start_time = time.time()

        while not all_done:
            if self.in_q.empty():
                print('[{0}] Worker {1} in q is empty... sleeping...'.format(
                    datetime.now().strftime("%H:%M:%S"), self.worker_idx
                ))
                time.sleep(2)
            else:
                incoming_value = self.in_q.get()
                if incoming_value == '<TERM>':
                    print('[{0}] Worker {1} rcvd <TERM> items...'.format(
                        datetime.now().strftime("%H:%M:%S"), self.worker_idx
                    ))
                    all_done = True
                    self.out_q.put('<TERM>')
                else:
                    try:
                        record_no = incoming_value
                        (line_tokens, line_gt_str_tokens, has_special_chars), line_source = self.generate_text(words_min, words_max)
                        line = ' '.join(line_tokens)
                        line_gt_str = ' \; '.join(line_gt_str_tokens)

                        selected_font = random.choice(self.available_font_names)
                        selected_fontsize = random.choices(self.fontsizes, self.fontsize_weights, k=1)[0]

                        base_filename = self.base_filename_tmplt.format(record_no)
                        valid_size = False
                        rcParams['text.usetex'] = has_special_chars
                        first_try = True
                        while not valid_size:
                            plt.text(0.0, 0.5, line, fontsize=selected_fontsize, fontname=selected_font)
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
                                if first_try:
                                    token_idx = random.randint(min(2, len(line_tokens) - 2), len(line_tokens) - 2)

                                    line_tokens[token_idx] = '{}\n{}'.format(line_tokens[token_idx], line_tokens[token_idx + 1])
                                    line_gt_str_tokens[token_idx] = '{} \\\\ {}'.format(
                                        line_gt_str_tokens[token_idx],
                                        line_gt_str_tokens[token_idx + 1])

                                    del line_gt_str_tokens[token_idx + 1]
                                    del line_tokens[token_idx + 1]
                                    line = ' '.join(line_tokens)
                                    line_gt_str = ' \; '.join(line_gt_str_tokens)
                                    first_try = False

                                else:
                                    selected_fontsize -= 3
                            else:
                                valid_size = True

                            if selected_fontsize < 4:
                                break

                        if valid_size:
                            out_fp = os.path.join(self.final_out, base_filename)
                            cropped.save(out_fp)
                            n_processed += 1

                            rec_font_d = {
                                'font': selected_font,
                                'fontsize': selected_fontsize,
                                'source': line_source,
                            }

                            self.out_q.put([record_no, line_gt_str, rec_font_d])

                            if n_processed % summary_every == 0 or n_processed == 1:
                                elapsed_time = time.time() - start_time
                                print('[{0}] Worker {1} made {2} records... avg {3:.4}s/record'.format(
                                    datetime.now().strftime("%H:%M:%S"), self.worker_idx, n_processed,
                                    elapsed_time / n_processed
                                ))

                    except RuntimeError as ex:
                        plt.clf()
                    except Exception as ex:
                        print(
                            '$$ [{0}] Worker {1} had unexpected error processing \'{2}\' w/ gt \'{3}\' from {4}\nex: {5} $$'.format(
                                datetime.now().strftime("%H:%M:%S"), self.worker_idx, line, line_gt_str, line_source, ex
                            ))
                        plt.clf()

        elapsed_time = time.time() - start_time
        print('[{0}] Worker {1} processed a total of {2} records...  avg {3:.4}s/record'.format(
            datetime.now().strftime("%H:%M:%S"), self.worker_idx, n_processed, elapsed_time / n_processed
        ))

    def generate_many_text_and_process(self, n, words_min=1, words_max=12):
        label_d = {}
        font_d = {}
        n_processed = 0
        summary_every = max(n // 20, 1)

        while n_processed < n:
            try:
                (line, line_gt_str, has_special_chars), line_source = self.generate_text(words_min, words_max)

                selected_font = random.choice(self.available_font_names)
                selected_fontsize = random.choices(self.fontsizes, self.fontsize_weights, k=1)[0]
                raw_image_saved = False

                base_filename = self.base_filename_tmplt.format(self.record_idx_start + n_processed)
                valid_size = False
                rcParams['text.usetex'] = has_special_chars
                while not valid_size:
                    plt.text(0.0, 0.5, line, fontsize=selected_fontsize, fontname=selected_font)
                    plt.axis('off')
                    plt.box(False)
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='png', bbox_inches='tight')
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
                n_processed += 1

                label_d[base_filename] = line_gt_str
                font_d[base_filename] = {
                    'font': selected_font,
                    'fontsize': selected_fontsize,
                    'source': line_source,
                }

                if n_processed % summary_every == 0 or n_processed == 1:
                    print('[{0}] Worker {1} made {2}/{3} records'.format(
                        datetime.now().strftime("%H:%M:%S"), self.worker_idx, n_processed, n)
                    )
            except RuntimeError as ex:
                print('** [{0}] Worker {1} error processing line \'{2}\' w/ gt \'{3}\' from {4} **'.format(
                    datetime.now().strftime("%H:%M:%S"), self.worker_idx, line, line_gt_str, line_source
                ))
                plt.clf()
            except Exception as ex:
                print('$$ [{0}] Worker {1} had unexpected error processing \'{2}\' w/ gt \'{3}\' from {4}\nex: {5} $$'.format(
                    datetime.now().strftime("%H:%M:%S"), self.worker_idx, line, line_gt_str, line_source, ex
                ))
                plt.clf()

        print('[{0}] Worker {1} made {2} records... pushing results to out_q'.format(
            datetime.now().strftime("%H:%M:%S"), self.worker_idx, n_processed)
        )
        self.out_q.put([self.worker_idx, label_d, font_d])

    def generate_text(self, words_min=1, words_max=12):
        sources = ['arxiv', 'pubmed', 'medrxiv']
        source_weights = [100, 120, 8]
        selected_source = random.choices(sources, source_weights, k=1)[0]
        record = self.generate_record(selected_source, words_min, words_max)

        return record, selected_source

    def make_superscript_subscript(self):
        script_type = random.choices(['alphanumeric', 'numeric_only', 'string_only'],
                                     weights=[5, 10, 10], k=1)[0]

        n_chars = random.randint(1, 6 if script_type != 'numeric_only' else 3)
        out = []
        for char_no in range(n_chars):
            if script_type == 'alphanumeric':
                char = random.choice(self.alphanumeric)
            elif script_type == 'string_only':
                char = random.choice(self.letters)
            else:
                char = str(random.randint(0, 9))

            out.append(char)

        # out = ''.join(out)
        if random.random() <= 0.5:
            out = [v.upper() if not v.startswith('\\') else v for v in out]

        return out

    def make_latex_str(self):
        latex_types = ['no_arg_latex_symbols', 'one_arg_latex_symbols']
        weights = [len(no_arg_latex_symbols), len(one_arg_latex_symbols)]
        type_to_make = random.choices(latex_types, weights=weights, k=1)[0]

        if type_to_make == 'no_arg_latex_symbols':
            latex_symbol = random.choice(no_arg_latex_symbols)
        else:
            latex_symbol = random.choice(one_arg_latex_symbols)
            if random.random() <= 0.5:
                supp = random.choice(self.letters)
            else:
                supp = str(random.randint(0, 100))

            # print('latex_symbol: {} supp: {}'.format(latex_symbol, supp))
            latex_symbol = latex_symbol.format('{', supp, '}')

        return latex_symbol

    def generate_record(self, record_source, words_min=1, words_max=12):
        if record_source == 'arxiv':
            sampled_text = self.sample_arxiv_text()
        elif record_source == 'pubmed':
            sampled_text = self.sample_pubmed_text()
        else:
            sampled_text = self.sample_chemarxiv_text()

        sampled_text = sampled_text.replace(' .', '.').replace(' ,', ',').replace(' :', ':').replace('[ ', '[')
        sampled_text = sampled_text.replace(' ]', ']').replace('( ', '(').replace(' )', ')').replace('\n', ' ')
        sampled_text = sampled_text.replace('%', '\%').replace('&', '\&').replace('$', '\$')
        pre_re_text = sampled_text[:]
        sampled_text = re.sub(r'((!$)\\[A-Za-z]+{.*}(!$))', r'$\1$', sampled_text)  # wrap latex notation in $
        sampled_text = re.sub(r'(\\[a-zA-Z]+)', r'\1 ', sampled_text)  # add white space after latex so it can be preserved later

        pre_bracket_text = sampled_text[:]
        unmatched_bracked_idxs = find_bad_brackets(sampled_text)
        if len(unmatched_bracked_idxs) > 0:
            for bad_bracked_idx in unmatched_bracked_idxs[::-1]:
                sampled_text = sampled_text[:bad_bracked_idx] + sampled_text[bad_bracked_idx + 1:]

        sampled_text_tokens = [v for v in sampled_text.split(' ') if v != '']

        sampled_len = min(random.randint(words_min, words_max), len(sampled_text_tokens))
        start_idx = random.randint(0, len(sampled_text_tokens) - sampled_len)
        selected_text_tokens = sampled_text_tokens[start_idx: start_idx + sampled_len]

        selected_text_labels = [convert_str_to_label_format(t) for t in selected_text_tokens]

        has_special_chars = '$' in sampled_text

        if random.random() <= self.superscript_p:
            token_idx = random.randint(0, len(selected_text_tokens) - 1)
            superscript_value_list = self.make_superscript_subscript()
            superscript_value = ' '.join(superscript_value_list)

            raw_text = selected_text_tokens[token_idx]
            raw_label = selected_text_labels[token_idx]

            new_text = '{}$^{}{}{}$'.format(raw_text, '{', superscript_value, '}')
            new_label = '{} ^ {} {} {}'.format(raw_label, '{', ' '.join([c for c in superscript_value_list]), '}')

            selected_text_tokens[token_idx] = new_text
            selected_text_labels[token_idx] = new_label

            has_special_chars = True

        if random.random() <= self.subscript_p:
            token_idx = random.randint(0, len(selected_text_tokens) - 1)
            superscript_value_list = self.make_superscript_subscript()
            superscript_value = ' '.join(superscript_value_list)

            raw_text = selected_text_tokens[token_idx]
            raw_label = selected_text_labels[token_idx]

            new_text = '{}$_{}{}{}$'.format(raw_text, '{', superscript_value, '}')
            new_label = '{} _ {} {} {}'.format(raw_label, '{', ' '.join([c for c in superscript_value_list]), '}')
            selected_text_tokens[token_idx] = new_text
            selected_text_labels[token_idx] = new_label

            has_special_chars = True

        if random.random() <= self.latex_insertion_p:
            latex_to_insert = self.make_latex_str()

            token_idx = random.randint(0, len(selected_text_tokens) - 1)
            raw_text = selected_text_tokens[token_idx]
            raw_label = selected_text_labels[token_idx]
            new_text = '{} ${}$'.format(raw_text, latex_to_insert.replace(' ', ''))
            new_label = '{} \; {}'.format(raw_label, latex_to_insert)

            selected_text_tokens[token_idx] = new_text
            selected_text_labels[token_idx] = new_label

            has_special_chars = True

        if random.random() < self.newline_p and len(selected_text_tokens) > 2:
            n_newline_insertions = random.choices(self.n_newline_options, weights=self.n_newline_weights, k=1)[0]
            n_newline_insertions = min(len(selected_text_tokens) - 1, n_newline_insertions)
            for _ in range(n_newline_insertions):
                token_idx = random.randint(0, len(selected_text_tokens) - 2)

                selected_text_tokens[token_idx] = '{}\n{}'.format(selected_text_tokens[token_idx],
                                                                  selected_text_tokens[token_idx + 1])
                selected_text_labels[token_idx] = '{} \\\\ {}'.format(selected_text_labels[token_idx],
                                                                      selected_text_labels[token_idx + 1])

                del selected_text_labels[token_idx + 1]
                del selected_text_tokens[token_idx + 1]

        return selected_text_tokens, selected_text_labels, has_special_chars

    def sample_pubmed_text(self):
        is_acceptable = False
        sampled_text = None
        bad_text = ['@xmath', '@xcite', '\\usepackage', '\\setlength', '\\setlength',
                    '\\end{document}', '\\begin{document}']
        while not is_acceptable:
            sampled_doc = random.choice(self.pubmed_documents)
            sampled_text = random.choice(sampled_doc['sections']).strip()

            if sampled_text != '' and not any(bt in sampled_text for bt in bad_text):
                is_acceptable = True

        return sampled_text

    def sample_arxiv_text(self):
        is_acceptable = False
        sampled_text = None
        while not is_acceptable:
            sampled_doc = random.choice(self.arxiv_documents)
            sampled_text = random.choice(sampled_doc['article_text']).strip()

            if sampled_text != '' and '@xmath' not in sampled_text and '@xcite' not in sampled_text:
                is_acceptable = True

        return sampled_text

    def sample_chemarxiv_text(self):
        is_acceptable = False
        sampled_text = None
        while not is_acceptable:
            sampled_doc = random.choice(self.chemrxiv_documents)
            sampled_text = sampled_doc['abstract']
            is_acceptable = True

        return sampled_text
