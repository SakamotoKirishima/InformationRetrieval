class LinkedList:
    def __init__(self):
        self._head = None

    def add_node(self, value):
        current_node = self._head
        if current_node is None:
            self._head = Node(value)
            return
        if current_node.get_value().doc_id > value.doc_id:
            node = Node(value)
            node.set_next(self._head)
            self._head = node
            return
        while current_node.get_next() is not None:
            if current_node.get_value().doc_id > value.doc_id:
                break
            current_node = current_node.get_next()
        node = Node(value)
        node.set_next(current_node.get_next())
        current_node.set_next(node)
        return

    def __str__(self):
        data = list()
        current_node = self._head
        while current_node is not None:
            data.append(str(current_node.get_value()))
            current_node = current_node.get_next()
        string = ' '
        string = string.join(data)
        return string


class Node:
    def __init__(self, value, previous=None):
        self._value = value
        # print self._value
        self._previous = previous
        self._next = None

    def set_next(self, next=None):
        self._next = next

    def get_next(self):
        return self._next

    def set_value(self, new_value):
        self._value = new_value

    def get_value(self):
        return self._value

    def set_previous(self, new_previous=None):
        self._previous = new_previous

    def get_previous(self):
        return self._previous

    def __str__(self):
        string = self._value
        return string
