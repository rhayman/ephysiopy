__version__ = '0.1.3'
import mahotas # this is to get around a weird, possibly python3.6 related issue
kk_path = '/home/robin/klustakwik/KlustaKwik'
'''
Check for the presence of the axona header files which may or may not be there
If not download them
'''
empty_headers = {
	"tetrode" : os.path.join(os.path.dirname(__file__), "tetrode_header.pkl"),
	"pos" : os.path.join(os.path.dirname(__file__), "pos_header.pkl"),
	"set" : os.path.join(os.path.dirname(__file__), "set_header.pkl"),
	"eeg" : os.path.join(os.path.dirname(__file__), "eeg_header.pkl"),
	"egf" : os.path.join(os.path.dirname(__file__), "egf_header.pkl")
}