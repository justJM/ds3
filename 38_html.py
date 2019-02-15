#%%
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_startendtag(self, tag, attrs):
        print('Encountered a start tag : ',tag)
    
    def handle_endtag(self, tag):
        print('Encountered a end tag : ',tag)

    def handle_data(self, data):
        print('Encountered some data : ',data)

parser = MyHTMLParser()
parser.feed('<html>')

#%%
import requests
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

html_doc = """
<html>
<head><title>The Dormouse's story</title></head>
<body>
    <p class="title"><b>The Dormouse's story</b></p>
    <p class="story">Once upon a time there we three little sisters; and their names were 
        <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>, 
        <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and 
        <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>; 
         and they liveed at the bottom of a well.</p>

<p class="story">...</p>
"""

soup = BeautifulSoup(html_doc,'html.parser')
soup