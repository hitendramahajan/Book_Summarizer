#!/usr/bin/python
import threading
import time
from utils.summarize import summarize
from utils.summarize import train_numbers
from utils.summarize import html_to_text
from utils.summarize import write_csv_to_pdf
from utils.summarize import merge_pdf
from scipy import integrate
from celery import Celery
import csv
import shutil
import os
import textwrap
import logging
import argparse
import sys
import pytest
import traceback
import tensorflow as tf
import numpy as np
import scipy as sc
import pandas as pd
import os
from pyspark.conf import SparkConf


def parse_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            A command line utility for website summarization.
            -----------------------------------------------
            These are common commands for this app.'''))
    parser.add_argument(
        'action',
        help='This action should be summarize')
    parser.add_argument(
        '--url',
        help='A link to the website url'
    )
    parser.add_argument(
        '--sentence',
        help='Argument to define number of sentence for the summary',
        type=int,
        default=2)
    parser.add_argument(
        '--language',
        help='Argument to define language of the summary',
        default='English')
    parser.add_argument(
        '--path',
        help='path to csv file')

    return parser.parse_args(argv[1:])

def square(x):
    return x ** 2

def readCsv(path):
    print('\n\n Processing Csv file \n\n')
    sys.stdout.flush()
    np.round_([.5, 1.5, 2.5, 3.5, 4.5, 10.1]) # type: ignore
    np.product([1, 2]) # type: ignore
    np.finfo(float)
    square(2)
    data = []
    try:
        with open(path, 'r') as userFile:
            userFileReader = csv.reader(userFile)
            for row in userFileReader:
                data.append(row)
    except:
        with open(path, 'r', encoding="mbcs") as userFile:
            userFileReader = csv.reader(userFile)
            for row in userFileReader:
                data.append(row)       
    return data


def writeCsv(data, LANGUAGE, SENTENCES_COUNT):
    print('\n\n Updating Csv file \n\n')
    sys.stdout.flush()
    with open('beneficiary.csv', 'w') as newFile:
        newFileWriter = csv.writer(newFile)
        length = len(data)
        position = data[0].index('website')
        for i in range(1, length):
            if i == 1:
                _data = data[0]
                _data.append("summary")
                newFileWriter.writerow(_data)
            try:
                __data = data[i]
                summary = summarize(
                    (data[i][position]), LANGUAGE, SENTENCES_COUNT)
                __data.append(summary)
                newFileWriter.writerow(__data)
            except:
                print('\n\n Error Skipping line \n\n')
                sys.stdout.flush()


def processCsv(path, LANGUAGE, SENTENCES_COUNT):
    try:
        print('\n\n Proessing Started \n\n')
        sys.stdout.flush()
        data = readCsv(path)
        writeCsv(data, LANGUAGE, SENTENCES_COUNT)
    except:
        print('\n\n Invalid file in file path \n\n')
        sys.stdout.flush()


def main(argv=sys.argv):
    identify_sequence()
    calculate_absolute_values()
    #write_csv_to_pdf()
    result = convert_categorical_to_list()
    merge_pdf()

    """
    # Configure logging
    logging.basicConfig(filename='applog.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(levelname)s:%(message)s')
    args = parse_args(argv)
    action = args.action
    url = args.url
    path = args.path
    LANGUAGE = "english" if args.language is None else args.language
    SENTENCES_COUNT = 2 if args.sentence is None else args.sentence
    
    if action == 'bulk':
        if path is None:
            print(
                '\n\n Invalid Entry!, please Ensure you enter a valid file path \n\n')
            sys.stdout.flush()
            return
        # guide against errors
        try:
            processCsv(path, LANGUAGE, SENTENCES_COUNT)
        except:
            print(
                '\n\n Invalid Entry!, please Ensure you enter a valid file path \n\n')
            sys.stdout.flush()
        print('Completed')
        sys.stdout.flush()
        if os.path.isfile('beneficiary.csv'):
            return shutil.move('beneficiary.csv', path)
        return
    if action == 'simple':
        # guide against errors
        try:
            html_to_text()
            summarize(url, LANGUAGE, SENTENCES_COUNT)
        except:
            print(
                '\n\n Invalid Entry!, please Ensure you enter a valid web link \n\n')
            sys.stdout.flush()
        print('Completed')
        sys.stdout.flush()
    else:
        print(
            '\nAction command is not supported\n for help: run python3 app.py -h'
        )
        sys.stdout.flush()
        return
    """
    

def identify_sequence():
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    integrate.simpson(y, x)
    y = np.power(x, 3)
    integrate.simpson(y, x)
    integrate.quad(lambda x: x**3, 0, 9)[0]


def create_thread():
    time.sleep(1)
    t = threading.Thread(target=create_thread)
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        print("Thread is still running...")
    else:
        print("Thread has finished")


def calculate_absolute_values():
    try:
        data1 = {'A': [-1, 2, -3], 'B': [-4, 5, -6]}
        df = pd.DataFrame(data1)
        result = df.applymap(abs)
        stack_dataframe(df)
    except Exception:
        traceback.print_exc()

        
def train_sequence():
    try:
        nested_structure = (1, 2, [3, 4])
        
        is_sequence = tf.nest.is_nested(nested_structure)
        print(is_sequence)
    except Exception:
        traceback.print_exc()
    

def convert_categorical_to_list():
    categories = pd.Categorical(["A", "B", "C", "A", "B"])
    cat_list = categories.to_list()
    return cat_list



def stack_dataframe(df):
    stacked_df = df.stack()
    return stacked_df


def getConf():
    conf = SparkConf()
    conf.setMaster("local").setAppName("Web Page Summation")
    conf.setSparkHome("/path")
    return conf

def get_broker_host():
    broker_host = os.getenv('BROKER_HOST')
    app = Celery('myapp', broker_url=broker_host)
    broker_port = os.getenv('BROKER_PORT')
    app = Celery('myapp', broker_url=f'amqp://localhost:{broker_port}')

    
if __name__ == '__main__':  
    main()
    train_sequence()
   
    
    
