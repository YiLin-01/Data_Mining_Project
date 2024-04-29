from itertools import compress


class MaskableList(list):
    '''
    if ml is MaskableList object，
    mask is a bool list, then ml[mask] return a new MaskableList object，
    which only include the True element in the mask list
    '''
    def __getitem__(self, index):
        try:
            # return super(MaskableList, self).__getitem__(index)
            return super().__getitem__(index)
        except TypeError:
            return MaskableList(compress(self, index))
