# PEaCE

Code to recreate PEaCE dataset for LREC-COLING 2024 paper [PEaCE: A Chemistry-Oriented Dataset for Optical Character Recognition on Scientific Documents](). 

# Obtain Existing PEaCE Dataset
The original dataset of 1M+ records can be found [here](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/njz5124_psu_edu/ESmEFZMuTK5EnQ2sHLWvDs8BWosWkCHUEvgeQCcdIJq8LA?e=qDMfty).
The file can be de-compressed by executing the following command: `tar -xzvf PEaCE.tar.gz`. 
The file is structured as follows:
- PEaCE
  - final_renders/
  - train.txt
  - dev.txt
  - test.txt
  - fonts.jsonl
  - labels.jsonl
  - vocab.jsonl

Each line of `[train|dev|test].txt` contains the filename of one record in the `final_renders` directory. 
The splits contain a roughly equal distribution of printed English, chemical equation, and numerical records. 
`fonts.jsonl` describes the fontsize and font type used to render each record.
Each `(token, count)` pair in `vocab.jsonl` describes a token in the vocabulary of the OCR models we explored and their count in the dataset. 

The real-world test records can be found in the `data/real_world_test_set` directory in this repository.

# Creating a New Dataset
Running ``make_records.py`` as given will create a dataset using the same parameters as described in the publication. 
We provide a brief description of important parameters below to enable others to create datasets suitable for their applications.

## Downloading Printed English Source Data

### ArXiv and PubMed
The ArXiv and PubMed datasets can be found in Arman Cohan's [``long-summarization``](https://github.com/armancohan/long-summarization) repository. 
Create `arxiv-dataset` and `pubmed-dataset` directories in `data/`, placing the corresponding `[train|val|test].txt` files in each directory.

### chemRxiv
The chemRxiv data can be obtained using the `chemrxiv()` method provided in the [`paperscraper`](https://github.com/PhosphorylatedRabbits/paperscraper) repository. 
The data (a `jsonl` file) will be downloaded to the `server_dumps` folder in the `paperscraper` repo. Create a `chemrxiv-dataset` directory in `data/`, and place the downloaded `jsonl` file there.


## Creating New Records

### Structural Parameters
- `--n_[text|chemeq|numeral]_sample`: The number of printex english, chemical equation, and numeric records to create.
- `--text_[superscript|subscript]_p`: The probability ([0, 1]) of a superscript/subscript being added to a printed English record.
- `--text_latex_insertion_p`: The probability ([0, 1]) of a latex symbol being randomly inserted in a printed English record.
- `--text_newline_p`: The probability ([0, 1]) of carriage returns being randomly inserted in a printed English record.
- `--text_[min|max]_n_words`: The minimum/maximum number of words to be included in a printed English record.
- `--chemeq_n_compound`: The maximum number of compounds to be included in each chemical equation record.
- `--chemeq_n_elements`: The maximum number of elements to be included in each compound in a chemical equation record. 
- `--chemeq_n_quanitty`: The maximum quantity associated with any element in a chemical equation record.
- `--numeral_symbol_p`: The probability ([0, 1]) that a LaTex symbol be generated in place of a number in numerical records.
- `--numeral_decimal_p`: The probability ([0, 1]) of a decimal being generated when creating a number for a numerical record. Else, an integer is created.
- `--numeral_max_numerals`: The maximum number of numbers/symbols to be included in numerical records. 
- `--max_w`: The maximum width, in pixels, of each constructed record.

### Stylistic Parameters
- `--fontsizes`: The font sizes that will be sampled when rendering each record.
- `--fontsize_weights`: The weight associated with sampling each fontsize.
- `--available_font_names`: The font types that will be samples when rendering each record.
  - We found [this](https://jonathansoma.com/lede/data-studio/matplotlib/list-all-fonts-available-in-matplotlib-plus-samples/) resource useful in identifying which fonts are available.

### Processing Parameters
- `n_workers`: The number of processes that will be used to construct records. 

An example of how a custom version of the dataset can be created is given below:
```python
python3 make_records.py --n_text_sample 50000 --n_chemeq_sample 2500 --n_numeral_sample 1500 \
    --text_superscript_p 0.05 --text_subscript_p 0.025 --text_latex_insertion_p 0.1 \
    --chemeq_n_compound 6 --numeral_max_numerals 3 \
    --fontsizes 12 20 32 --fonsize_weights 1 2 3 --available_font_names 'Noto Mono' 'URW Bookman'
```

The above command creates 50,000 printed English records, 2,500 chemical equation records, and 1,500 numerical records. 
Superscripts are added to each record with 5% probability, subscripts with 2.5% probability, and LaTex symbols with 10% probability. 
Each chemical equation will contain upto 6 compound, and each numerical record will contain upto 3 numerals/LaTex symbols. 
Records will be rendered with either `Noto Mono` or `URW Bookman` in one of three fontsizes, with the larger fontsizes having a larger weight for sampling. 

## Citation
