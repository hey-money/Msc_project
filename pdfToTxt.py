import PyPDF2
import os

directory = os.path.abspath(os.path.dirname(__file__))
rawWhitePapers = directory + "/rawWhitePapers"
txtWhitePapers = directory + "/txtWhitePapers"

if not os.path.exists(txtWhitePapers):
    os.makedirs(txtWhitePapers)
    
    
for filename in os.listdir(rawWhitePapers):
    print(filename)
    if filename == ".DS_Store":
        pass
    else:
        pdffileobj = open(directory + "/rawWhitePapers" + "/" + filename, 'rb')
        # create reader variable that will read the pdffileobj
        pdfreader = PyPDF2.PdfFileReader(pdffileobj)
        # This will store the number of pages of this pdf file
        x = pdfreader.numPages
        for page in range(x):
            pageobj = pdfreader.getPage(page)
            text = pageobj.extractText()
            txt_filepath = os.path.join(txtWhitePapers, f"{os.path.splitext(filename)[0]}.txt")
            
            with open(txt_filepath, 'a', encoding='utf-8') as file1: 
                file1.write(text)
            file1.close()
