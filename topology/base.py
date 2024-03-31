from torch.utils.tensorboard import SummaryWriter


class ParentContainer:
    def __init__(self, parent: ['TopologyBase', None]):
        self.obj = parent


class TopologyBase:
    def __init__(self, tag: str = '', parent: 'TopologyBase' = None, writer: SummaryWriter = None, layers: list['TopologyBase'] = ()):
        self.step = 0
        self.tag = tag
        self.writer = writer
        self._parent = ParentContainer(parent)
        self.topology_children: dict[int, 'TopologyBase'] = {id(x): x for x in layers}

    def parent(self):
        return self._parent.obj

    def set_parent(self, parent: ['TopologyBase', None]):
        self._parent.obj = parent

    def get_writer(self):
        if self.parent() is not None:
            return self.writer or self.parent().get_writer()
        return self.writer

    def get_tags(self):
        if self.parent() is not None:
            return self.parent().get_tags() + [self.tag + f' (Call {self.step})']
        return [self.tag + f' (Call {self.step})']

    def flush(self):
        self.step = 0
        for k in self.topology_children:
            self.topology_children[k].flush()
