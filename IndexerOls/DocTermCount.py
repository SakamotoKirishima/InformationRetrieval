class DocTermCount:
    def __init__(self, doc_id, count):
        self.doc_id = doc_id
        self.count = count

    def __str__(self):
        return 'DocID= ' + str(self.doc_id) + '\t count= ' + str(self.count)
