
class List_Queue:

    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)

if __name__ == '__main__':
    # init
    queue = List_Queue()
    queue.push(3)
    queue.push(8)
    print(queue.size())
    print(queue.pop())
