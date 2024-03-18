import os
import re
import math
import copy
import json
import time
import argparse

from datetime import date, datetime
from multiprocessing import Process, Manager

from src import PrintedEnglishMaker, ChemEqNumeralMaker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--arxiv_data1', default='/home/czh/md0/datasets/arxiv-dataset/test.txt')
    parser.add_argument('--arxiv_data2', default='/home/czh/md0/datasets/arxiv-dataset/val.txt')

    parser.add_argument('--pubmed_data', default='/home/czh/md0/datasets/pubmed-dataset/test.txt')
    parser.add_argument('--chemrxiv_data', default='/home/czh/md0/datasets/chemrxiv_dataset/chemrxiv_2023-06-01.jsonl')

    parser.add_argument('--out', default='created_records/')

    parser.add_argument('--n_text_sample', default=1000, type=int)  # , default=1000000
    parser.add_argument('--n_chemeq_sample', default=100, type=int)  # , default=100000
    parser.add_argument('--n_numeral_sample', default=100, type=int)  # , default=100000

    parser.add_argument('--text_superscript_p', default=0.0375, type=float)
    parser.add_argument('--text_subscript_p', default=0.0125, type=float)
    parser.add_argument('--text_latex_insertion_p', default=0.15, type=float)
    parser.add_argument('--text_newline_p', default=0.15, type=float)
    parser.add_argument('--text_min_n_words', default=1, type=int)
    parser.add_argument('--text_max_n_words', default=10, type=int)

    parser.add_argument('--chemeq_n_compound', default=4, type=int)
    parser.add_argument('--chemeq_n_elements', default=4, type=int)
    parser.add_argument('--chemeq_n_quantity', default=500, type=int)

    parser.add_argument('--numeral_symbol_p', default=0.1, type=float)
    parser.add_argument('--numeral_decimal_p', default=0.5, type=float)
    parser.add_argument('--numeral_max_numerals', default=4, type=int)

    parser.add_argument('--fontsizes', default=[12, 16, 20, 24, 28, 32], type=int, nargs='+')
    parser.add_argument('--fontsize_weights', default=[5, 10, 10, 10, 7.5, 2.5], type=float, nargs='+')
    parser.add_argument(
        '--available_font_names',
        default=[
            'Noto Mono', 'Noto Serif Display', 'FreeSans',
            'FreeMono', 'Liberation Mono',
            'Noto Sans CJK JP', 'Noto Sans Math',
            'URW Bookman',
        ],
        type=str, nargs='+')

    parser.add_argument('--n_workers', default=2, type=int)    # , default=16
    parser.add_argument('--max_w', default=600, type=int)

    args = parser.parse_args()

    input('args.available_font_names: {}'.format(args.available_font_names))

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    final_out = os.path.join(args.out, 'final_renders')
    if not os.path.exists(final_out):
        os.makedirs(final_out)

    label_out_fp = os.path.join(args.out, 'labels.jsonl')
    font_out_fp = os.path.join(args.out, 'meta.jsonl')
    label_f = open(label_out_fp, 'w+')
    meta_f = open(font_out_fp, 'w+')

    base_filename_tmplt = 'render_{}.png'
    final_filename_tmplt = os.path.join(final_out, base_filename_tmplt)

    available_font_names = args.available_font_names
    fontsizes = args.fontsizes
    fontsize_weights = args.fontsize_weights
    n_ingest = 0

    if args.n_text_sample > 0:
        print('Reading arxiv_documents...')
        arxiv_documents = []
        arxiv_n_words = 0
        for jsonl_fp in [args.arxiv_data1, args.arxiv_data2]:
            with open(jsonl_fp, 'r') as f:
                for line in f:
                    j = json.loads(line.strip())
                    arxiv_n_words += len(' '.join(j['article_text']).split(' '))
                    arxiv_documents.append(j)

        print('\tlen(arxiv_documents): {}'.format(len(arxiv_documents)))
        print('\tarxiv_n_words: {}'.format(arxiv_n_words))

        print('Reading pubmed_documents...')
        pubmed_documents = []
        pubmed_n_words = 0
        with open(args.pubmed_data, 'r') as f:
            for line in f:
                j = json.loads(line.strip())
                j['sections'] = [' '.join([ss.capitalize() for ss in s]) for s in j['sections']]
                pubmed_n_words += len(' '.join(j['sections']).split(' '))
                # input("j['sections']: {}".format(j['sections']))
                pubmed_documents.append(j)
        print('\tlen(pubmed_documents): {}'.format(len(pubmed_documents)))
        print('\tpubmed_n_words: {}'.format(pubmed_n_words))

        print('Reading chemrxiv_documents...')
        chemrxiv_documents = []
        chemrxiv_n_words = 0
        CLEANR = re.compile('<.*?>')
        n_bad = 0
        with open(args.chemrxiv_data, 'r') as f:
            for line in f:
                j = json.loads(line.strip())
                try:
                    j['abstract'] = j['abstract'].encode('latin1').decode('unicode_escape').replace('\n', ' ')
                    j['abstract'] = re.sub(CLEANR, '', j['abstract'])
                    chemrxiv_documents.append(j)
                    chemrxiv_n_words += len(j['abstract'].split(' '))
                except Exception as ex:
                    n_bad += 1
        print('\tlen(chemrxiv_documents): {}'.format(len(chemrxiv_documents)))
        print('\tchemrxiv_n_words: {}'.format(chemrxiv_n_words))

        n_items_per_process = int(math.ceil(args.n_text_sample / args.n_workers))
        total_items = n_items_per_process * args.n_workers
        m = Manager()
        result_q = m.Queue()

        worker_in_qs = [m.Queue() for _ in range(args.n_workers)]

        printed_english_makers = [
            PrintedEnglishMaker(
                args,
                copy.deepcopy(arxiv_documents), copy.deepcopy(pubmed_documents),
                copy.deepcopy(chemrxiv_documents),
                outdir=args.out, superscript_p=args.text_superscript_p, subscript_p=args.text_subscript_p,
                newline_p=args.text_newline_p,
                record_idx_start=idx * n_items_per_process, out_q=result_q, worker_idx=idx,
                available_font_names=available_font_names, fontsizes=fontsizes, fontsize_weights=fontsize_weights,
                max_w=args.max_w, in_q=worker_in_qs[idx]
            ) for idx in range(args.n_workers)
        ]

        worker_procs = [
            Process(target=w.generate_and_process_from_q,
                    args=(args.text_min_n_words, args.text_max_n_words, max(n_items_per_process // 1000, 1)))
            for w in printed_english_makers
        ]

        curr_record_number = 0
        orchestration_start_time = time.time()
        print('Pre-loading worker qs...')
        n_items_left = args.n_text_sample
        for worker_in_q in worker_in_qs:
            n_to_put = min(args.n_text_sample - curr_record_number, 20)
            for record_no in range(curr_record_number, curr_record_number + n_to_put):
                worker_in_q.put(record_no)

            curr_record_number += n_to_put

        print('Starting {} worker processes...'.format(args.n_workers))
        for p in worker_procs:
            p.start()

        put_update_every = max(args.n_text_sample // 1000, 1)
        ingest_update_every = max(args.n_text_sample // 1000, 1)
        last_put_update = None
        last_ingest_update = None
        n_term_rcvd = 0

        start_time = time.time()
        while curr_record_number < args.n_text_sample:
            for worker_in_q in worker_in_qs:
                if worker_in_q.empty():
                    n_to_put = min(args.n_text_sample - curr_record_number, 20)
                    for record_no in range(curr_record_number, curr_record_number + n_to_put):
                        worker_in_q.put(record_no)

                    curr_record_number += n_to_put

            n_pushed = curr_record_number
            summary_ordinal = n_pushed // put_update_every
            if summary_ordinal != last_put_update:
                elapsed_time = time.time() - start_time
                print('[{0}] Orchestrator pushed {1} of {2} items to workers... avg {3:.2f}s/item...'.format(
                    datetime.now().strftime("%H:%M:%S"), n_pushed, args.n_text_sample, elapsed_time / n_pushed
                ))
                last_put_update = summary_ordinal

            while not result_q.empty():
                res = result_q.get()

                if res != '<TERM>':
                    record_no, record_label_d, record_font_d = res
                    record_filename = base_filename_tmplt.format(record_no)

                    record_label_d = {record_filename: record_label_d}
                    record_font_d = {record_filename: record_font_d}
                    label_f.write('{}\n'.format(record_label_d))
                    meta_f.write('{}\n'.format(record_font_d))

                    n_ingest += 1
                elif res == '<TERM>':
                    n_term_rcvd += 1
                    print('[{0}] Orchestrator received {1} or {2} <TERM> signals...'.format(
                        datetime.now().strftime("%H:%M:%S"), n_term_rcvd, args.n_workers
                    ))

                summary_ordinal = n_ingest // ingest_update_every
                if summary_ordinal != last_ingest_update:
                    elapsed_time = time.time() - start_time
                    print('[{0}] Orchestrator ingested {1} records... avg {2:.2f}s/record...'.format(
                        datetime.now().strftime("%H:%M:%S"), n_ingest, elapsed_time / n_ingest
                    ))
                    last_ingest_update = summary_ordinal

        print('[{0}] Orchestrator pushed {1} items to queue, pushing <TERM> items...'.format(
            datetime.now().strftime("%H:%M:%S"), args.n_text_sample
        ))
        for worker_in_q in worker_in_qs:
            worker_in_q.put('<TERM>')

        while n_term_rcvd < args.n_workers:
            if result_q.empty():
                print('[{0}] Orchestrator waiting for results... sleeping...'.format(
                    datetime.now().strftime("%H:%M:%S")
                ))
                time.sleep(10)
            else:

                labels_to_write = []
                fonts_to_write = []
                while not result_q.empty():
                    res = result_q.get()

                    if res == '<TERM>':
                        n_term_rcvd += 1
                        print('[{0}] Orchestrator received {1} of {2} <TERM> signals...'.format(
                            datetime.now().strftime("%H:%M:%S"), n_term_rcvd, args.n_workers
                        ))
                    else:
                        record_no, record_label_d, record_font_d = res
                        record_filename = base_filename_tmplt.format(record_no)
                        record_label_d = {record_filename: record_label_d}
                        record_font_d = {record_filename: record_font_d}
                        labels_to_write.append(str(record_label_d))
                        fonts_to_write.append(str(record_font_d))

                        n_ingest += 1

                        summary_ordinal = n_ingest // ingest_update_every
                        if summary_ordinal != last_ingest_update:
                            elapsed_time = time.time() - start_time
                            print('[{0}] Orchestrator ingested {1} records... avg {2:.2f}s/record...'.format(
                                datetime.now().strftime("%H:%M:%S"), n_ingest, elapsed_time / n_ingest
                            ))
                            last_ingest_update = summary_ordinal

                labels_to_write = '\n'.join(labels_to_write)
                fonts_to_write = '\n'.join(fonts_to_write)
                label_f.write('{}\n'.format(labels_to_write))
                meta_f.write('{}\n'.format(fonts_to_write))

    curr_idx = n_ingest
    if args.n_chemeq_sample > 0:
        print('Creating {} workers to make {} numeral records...'.format(args.n_workers, args.n_chemeq_sample))
        n_items_per_process = int(math.ceil(args.n_chemeq_sample / args.n_workers))
        total_items = n_items_per_process * args.n_workers
        m = Manager()
        result_q = m.Queue()

        generators = [
            ChemEqNumeralMaker(
                'chemeq', args.out,
                record_idx_start=curr_idx + (idx * n_items_per_process), out_q=result_q, worker_idx=idx,
                available_font_names=available_font_names, fontsizes=fontsizes, fontsize_weights=fontsize_weights,
                chemeq_n_compound=args.chemeq_n_compound, chemeq_n_elements=args.chemeq_n_elements,
                chemeq_n_quantity=args.chemeq_n_quantity,
                max_w=args.max_w
            ) for idx in range(args.n_workers)
        ]

        worker_procs = [
            Process(target=w.generate_many_record_and_process,
                    args=(n_items_per_process,))
            for w in generators
        ]

        print('Starting {} constructor processes...'.format(args.n_workers))
        for p in worker_procs:
            p.start()

        for p in worker_procs:
            p.join()

        while any(p.is_alive() for p in worker_procs):
            print('Sleeping!')
            time.sleep(30)

        worker_results = []
        while not result_q.empty():
            worker_idx, worker_label_d, worker_font_d = result_q.get()
            worker_results.append([worker_idx, worker_label_d, worker_font_d])
        worker_results = list(sorted(worker_results, key=lambda x: x[0]))
        for worker_idx, worker_label_d, worker_font_d in worker_results:
            print('Aggregating results from worker {}...'.format(worker_idx))
            for k, v in worker_label_d.items():
                # label_d[k] = v
                label_f.write('{}\n'.format({k: v}))
                curr_idx += 1

            for k, v in worker_font_d.items():
                # font_d[k] = v
                meta_f.write('{}\n'.format({k: v}))

    if args.n_numeral_sample > 0:
        print('Creating {} workers to make {} numeral records...'.format(args.n_workers, args.n_numeral_sample))
        n_items_per_process = int(math.ceil(args.n_numeral_sample / args.n_workers))
        total_items = n_items_per_process * args.n_workers
        m = Manager()
        result_q = m.Queue()

        generators = [
            ChemEqNumeralMaker(
                'numeral', args.out,
                record_idx_start=curr_idx + (idx * n_items_per_process), out_q=result_q, worker_idx=idx,
                available_font_names=available_font_names, fontsizes=fontsizes, fontsize_weights=fontsize_weights,
                numeral_decimal_p=args.numeral_decimal_p, numeral_max_numerals=args.numeral_max_numerals,
                numeral_symbol_p=args.numeral_symbol_p,
                max_w=args.max_w,
            ) for idx in range(args.n_workers)
        ]

        worker_procs = [
            Process(target=w.generate_many_record_and_process,
                    args=(n_items_per_process,))
            for w in generators
        ]

        print('Starting {} constructor processes...'.format(args.n_workers))
        for p in worker_procs:
            p.start()

        for p in worker_procs:
            p.join()

        while any(p.is_alive() for p in worker_procs):
            print('Sleeping!')
            time.sleep(30)

        worker_results = []
        while not result_q.empty():
            worker_idx, worker_label_d, worker_font_d = result_q.get()
            worker_results.append([worker_idx, worker_label_d, worker_font_d])
        worker_results = list(sorted(worker_results, key=lambda x: x[0]))
        for worker_idx, worker_label_d, worker_font_d in worker_results:
            print('Aggregating results from worker {}...'.format(worker_idx))
            for k, v in worker_label_d.items():
                label_f.write('{}\n'.format({k: v}))
                curr_idx += 1

            for k, v in worker_font_d.items():
                meta_f.write('{}\n'.format({k: v}))

    print('all done :)')
