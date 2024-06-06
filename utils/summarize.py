from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import sys
import numpy as np
from scipy import integrate
import html2text
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from PyPDF2 import PdfFileMerger
from PyPDF2 import PdfMerger
from PIL import Image, ImageFont
from PyPDF2 import PdfReader
from PyPDF2 import PdfWriter
import os


def html_to_text():
    html_text = "<p>Web Page Summation.</p>"
    text_converter = html2text.html2text(html_text)
    return text_converter


def write_csv_to_pdf():
    ldr = os.getcwd()+r"\test.pdf"
    writer = PdfFileWriter()
    with open(ldr, "rb") as f:
        reader = PdfFileReader(f)
        page = reader.getPage(0)
        writer.addPage(page)

def merge_pdf():
    merger = PdfFileMerger() 
    print(merger.id_count) 
    merger.close()

       
def summarize(url=None, LANGUAGE='English', SENTENCES_COUNT=2):
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    result = ''
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        result = result + ' ' + str(sentence)
        try:
            result = result + ' ' + str(sentence)

        except:
            print(
                    '\n\n Invalid Entry!, please Ensure you enter a valid web link \n\n')
            sys.stdout.flush()
            return (
                    '\n\n Invalid Entry!, please Ensure you enter a valid web link \n\n')
    print('\n\n'+str(url)+'\n\n'+str(result))
    sys.stdout.flush()
    return result

def train_numbers():
    arrid = np.array([2,3,4,4])
    product_id = np.product(arrid)
    np.round(arrid)
    np.cumprod(arrid)
    return product_id
 
    
