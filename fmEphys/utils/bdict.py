class AnyBase(type):
    """
    """
    def __eq__(self, other):
        return True

    def __repr__(self):
        return 'Any'

    @classmethod
    def __subclasscheck__(cls, other):
        return True

    @classmethod
    def __instancecheck__(cls, other):
        return True

class Any(object):
    """
    https://stackoverflow.com/questions/29866269/searching-for-a-partial-match-in-a-list-of-tuples
    """
    __metaclass__ = AnyBase

    def __init__(self):
        raise NotImplementedError("How'd you instantiate Any?")


class bdict(dict):
    """
    Better dict with some helpful indexing tools
    """
    def __init__(self, data_in):
        """ Initialize with an existing dictionary, which can be nested.
        """
        self.d = data_in

    
    
    
    
#     def _key_match(self, target, l):
#         """
#         key, value pattern
#         e.g., 
#         """
#         for k, v in self.d.items():
#             if self._key_match(k, v, '*'):
        
        
#         try:
#           if fnmatch.fnmatch(k, pat):
#             return isinstance(v, dict)
#         except TypeError:
#             return False
    
    def __repr__(self):
        return repr(self.d)

    def _access(self, _filt_d, key_list):
        """
        https://stackoverflow.com/questions/49482258/getting-an-item-from-a-nested-dictionary-using-a-list-of-indexes-python
        """
        if len(key_list) == 0:
            return self.d
        return self._access(self.d[key_list[0]], key_list[1:])

    def _assign(self, key_list, insert_val)

    def __getitem__(self, ind):
        """
        expects literal match. can be int, string, etc. no positional indexing
        
        t is a tuple of any length e.g., (3, 2, *, 1)
        any position in the tuple can be of type `Any`, and that
        level of the Fdict will be left 'wild'
        """

        


        if isinstance(ind, tuple)==False:
            ind = tuple(ind)
        
        print [v for v in self.d.keys() if match(ind, v)]



