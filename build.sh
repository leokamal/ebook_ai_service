pip3 install --upgrade pysqlite3-binary
pip3 install --upgrade langchain

package__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')