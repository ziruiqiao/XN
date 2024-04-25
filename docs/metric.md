For a picture of a bookshelf, the first step in trying to determine the accuracy of
the api and algorithm is to find the names of all the books by the human eye(***Human Eye Book Name List***),
Then, applied API and Algorithm to get all names of books by Computing(***Detected Book Name List***).
Finally, compare the two list of book names to get a ***confusion matrix***.
### OCR Confusion Matrix
Expect OCR to return only all the text that appears in the image, not all the book names

1. Matched By Both: Text exists in both ***Human Eye Book Name List*** and ***Detected Book Name List***
2. Only by OCR: Text exists in only ***Detected Book Name List*** but not ***Human Eye Book Name List***
3. OCR Book Name Total Amount: Total number of Text in ***Detected Book Name List***
4. Only by Human: Text exists in only ***Human Eye Book Name List*** but not ***Detected Book Name List***
5. Human Eye Book Name Total Amount: Total number of Text in ***Human Eye Book Name List***
### Text Group Confusion Matrix
Expect algorithm to only group book names without sorting

1. Matched By Both: Book Names exists in both ***Human Eye Book Name List*** and ***Detected Book Name List***
2. Only by Algorithm: Book Names exists in only ***Detected Book Name List*** but not ***Human Eye Book Name List***
3. Algorithm Book Name Total Amount: Total number of Book Names in ***Detected Book Name List***
4. Only by Human: Book Names exists in only ***Human Eye Book Name List*** but not ***Detected Book Name List***
5. Human Eye Book Name Total Amount: Total number of Book Names in ***Human Eye Book Name List***

Recall: ***Matched By Both*** / number of ***Detected Book Name List***
Precision: ***Matched By Both*** / number of ***Human Eye Book Name List***