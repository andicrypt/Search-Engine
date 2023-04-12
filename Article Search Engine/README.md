# Article Search Engine

* The VnExpress articles are crawled recursively by BeautifulSoup with arbitraty number of level of crawling. 
* From the crawled data, Whoosh library is used to index those artical (document) and to support later search task. 
* Furthermore,Streamlit library is used to make user-frinedly GUI for this engine.
<br>

## Source code includes:

* main.ipynb: run all cells in this file, the resulted application link is printed in the last cell.
* requirements.txt: as working environemnt is Colab, it is unnecessary to install by using requirements.txt since all packages are installed by running cell already.


***Note***: As I use Colab, I need to use Ngrok to connect my local host to internet host. Sometimes Ngrok's service is overloaded, it may result in error in the last cell. However, it rarely happens. If it happens, disconnect and refresh session, and then rerun all cells again. 

***Note***: Currently I set level of crawl is 2 in order to retrieve less data faster. However, I can set level of crawl to higher value by changing the max_nlevel param.


