### Tkinter GUI
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkFont
from tkinter import messagebox, filedialog
import sys, os
import io, yaml
import pandas as pd
import numpy as np
from pandastable import Table, TableModel

class PandasFrame(tk.Frame):
        """Basic test frame for the table"""
        def __init__(self, parent=None):
            self.parent = parent
            Frame.__init__(self)
            self.main = self.master
            self.main.geometry('600x400+200+100')
            self.main.title('Table app')
            f = Frame(self.main)
            f.pack(fill=BOTH,expand=1)
            df = TableModel.getSampleData()
            self.table = pt = Table(f, dataframe=df,
                                    showtoolbar=True, showstatusbar=True)
            pt.show()
            return

"""
class DataBase(object):
    def __init__(self):
        self.data = {'experiments': [],
                         'videos': [],
                         'session': [],
                         'dataset': []}

    def add_experiment(self, name, date, status, **kwargs):
        self.data['experiments'].append(name)
        entry = {}
        entry['id'] = "{:05d}".format(len(self.data['experiments']) - 1)
        entry['name'] = name
        entry['date'] = date
        entry['status'] = status
        ### extra entries
        for kw,v in kwargs.items():
            entry[kw] = v

        with open('./data/experiments/{}_{}.yaml'.format(entry['id'], entry['name']), 'w', encoding='utf8') as f:
            yaml.dump(entry, f, default_flow_style=False, allow_unicode=True, canonical=False)

    def filter(self, val):
        self.view = self.data.copy()
        for k, v in self.data.items():
            if val in k or val in v:
                pass
            else:
                self.view = self.view.pop(k)

"""

d = [ 'Name',
      '# males',
      '# females',
      'Age males',
      'Age females',
      'Genotype',
      'Stimulus',
      'Temperature']

### This is for window
class App():
    def __init__ (self):
        self.Vars = []
        self.Categories = []
        self.listFlies = []
        self.root = self.makeWindow()
        self.setSelect ()
        self.root.mainloop()

    def makeWindow (self):
        root = Tk()
        root.title("Fly Logger v0.1")
        root.resizable(width=True, height=True)
        root.geometry("400x600")

        # create a menu
        menu = Menu(root)
        root.config(menu=menu)

        filemenu = Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="New", command=self.restart)
        filemenu.add_command(label="Open...", command=self.open_file)
        filemenu.add_command(label="Save...")
        filemenu.add_separator()
        filemenu.add_command(label="Import...")
        filemenu.add_command(label="Export...")
        filemenu.add_separator()
        filemenu.add_command(label="Exit")

        viewmenu = Menu(menu)
        menu.add_cascade(label="View", menu=viewmenu)
        viewmenu.add_command(label="View experiment")
        viewmenu.add_command(label="View history")

        helpmenu = Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...")

        frame1 = Frame(root)
        frame1.pack()

        Label(frame1, text="Please type in information about experimental run").grid(row=0, column = 0,  columnspan=2, sticky=tk.W+tk.E, pady=2)
        for ind, cats in enumerate(d):
            self.addCategory(frame1, cats, ind)

        frame2 = Frame(root)       # Row of buttons
        frame2.pack()
        b1 = Button(frame2,text=" Add  ",command=self.addEntry)
        #b2 = Button(frame2,text=" Load ",command=loadEntry)
        b3 = Button(frame2,text="Delete",command=self.deleteEntry)
        b4 = Button(frame2,text=" Edit ",command=self.updateEntry)
        b1.pack(side=LEFT)
        b3.pack(side=LEFT)
        b4.pack(side=LEFT)

        frame3 = Frame(root)       # select of names
        frame3.pack()
        scroll = Scrollbar(frame3, orient=VERTICAL)
        self.select = Listbox(frame3, yscrollcommand=scroll.set, height=12)
        self.select.bind('<Double-Button-1>', self.doubleClicked)
        scroll.config (command=self.select.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.select.pack(side=LEFT,  fill=BOTH, expand=1)

        frame4 = Frame(root)
        frame4.pack()


        return root

    def restart (self):
        self.listFlies = []
        self.Vars = []
        self.Categories = []
        self.setSelect()
        self.root.destroy()
        self.root = self.makeWindow()

    def open_file(self):
        name= filedialog.askopenfilename()
        print(name)

    def save_file(self):
        print("Yeah")

    def doubleClicked (self, event) :
        self.loadEntry()

    def whichSelected (self) :
        if len(self.select.curselection()) > 0:
            return int(self.select.curselection()[0])
        else:
            print("Warning: No entry selected.")
            return None

    def addCategory (self, frame, title, ind):
        Label(frame, text=title).grid(row=ind+1, column=0, sticky=tk.W, pady=1)
        self.Vars.append(StringVar())
        self.Categories.append(Entry(frame, textvariable=self.Vars[-1]))
        self.Categories[-1].grid(row=ind+1, column=1, sticky=tk.W, pady=1)

    def addEntry (self):
        temp = []
        for var in self.Vars:
            temp.append(var.get())
        fly = Fly(temp)
        self.listFlies.append(fly)
        self.setSelect ()

    def updateEntry (self):
        for ind, var in enumerate(self.Vars):
            self.listFlies[self.whichSelected()].set(ind, var.get())
        setSelect ()

    def deleteEntry (self):
        if self.whichSelected() == None:
            print("")
        else:
            del self.listFlies[self.whichSelected()]
        self.setSelect ()

    def loadEntry (self):
        for ind, var in enumerate(self.Vars):
            temp = self.listFlies[self.whichSelected()].get(ind)
            var.set(temp)

    def setSelect (self) :
        self.select.delete(0,END)
        for ind, fly in enumerate(self.listFlies):
            self.select.insert (END, "{:03}".format(ind+1) + "\t\t" + fly.get(0))

class Fly:
    def __init__(self, indata):
        self.data = []
        for vals in indata:
            self.data.append(vals)

    def get(self, ind):
        return self.data[ind]

    def set(self, ind, val):
        self.data[ind] = val


class MenuBar(object):
    def __init__(self, frame, struct):
        self.menu = tk.Menu()
        frame.config(menu=self.menu)
        for cascade, command in struct.items():
            submenu = tk.Menu(self.menu)
            self.menu.add_cascade(label=cascade, menu=submenu)
            for label, func in command.items():
                if '&sep' in label: submenu.add_separator()
                else: submenu.add_command(label=label, command=func)

class NoteBook(ttk.Notebook):
    def __init__(self, frame, struct):
        ttk.Notebook.__init__(self)
        self.tabs = {}
        for k, v in struct.items():
            self.tabs[k] = tk.Frame(self)
            self.add(self.tabs[k], text=k)
        self.pack(fill=tk.BOTH,expand=1)


class BacktestWindow(object):
    def __init__(self, master):
        self.master = master

        self.dir = os.getcwd()
        self.fileList = []

        self.data = pd.DataFrame()
        self.currfile = None

        ### system adjustments
        if sys.platform == 'darwin':
            s = ttk.Style()
            s.configure('TNotebook', tabposition='nw')
            s.configure('TNotebook.Tab', padding=(20, 8, 20, 0))

        ### general
        self.master.title("Backtesting.py Simulation")
        self.master.resizable(width=True, height=True)
        self.master.geometry("800x600")

        ### menubar
        self.menubar = MenuBar(self.master, {   'File': {'Clear Data': self.clear_data, 'Open recent': None, 'Save Data': self.save_data, 'Save as...': self.saveas_db, '&sep1': None, 'Import': None, 'Export': None, '&sep2': None, 'Quit': self.master.destroy},
                                                #'Edit': {'Undo': None, 'Redo': None, '&sep1': None, 'Cut': None, 'Copy': None, 'Paste': None, '&sep2': None, 'Preferences': None},
                                                'View': {'Toggle fullscreen': None},
                                                'Window': {'Minimize': None},
                                                'Help': {'About': None}})

        ### notebook tabs
        self.notebook = NoteBook(self.master, { 'Fetch Data': None,
                                                'View Data': None,
                                                'Backtest': None,
                                                'Optimize': None,
                                                'Plot': None})

        ## tabs Widgets
        data_tab = self.notebook.tabs['Fetch Data']
        view_tab = self.notebook.tabs['View Data']
        bt_tab = self.notebook.tabs['Backtest']
        opt_tab = self.notebook.tabs['Optimize']
        plot_tab = self.notebook.tabs['Plot']

        ## Data Tab
        data_from_db = tk.Frame(data_tab)
        data_from_db.pack()

        tk.Label(data_from_db, text="Import Data from Database").pack()

        data_datepicker = tk.Frame(data_tab)       # Row of buttons
        frame2.pack()
        b1 = tk.Button(frame2,text=" Add  ",command=None)
        #b2 = Button(frame2,text=" Load ",command=loadEntry)
        b3 = tk.Button(frame2,text="Delete",command=None)
        b4 = tk.Button(frame2,text=" Edit ",command=None)
        b1.pack(side=tk.LEFT)
        b3.pack(side=tk.LEFT)
        b4.pack(side=tk.LEFT)

        frame3 = tk.Frame(data_tab)       # select of names
        frame3.pack()
        scroll = tk.Scrollbar(frame3, orient=tk.VERTICAL)
        self.select = tk.Listbox(frame3, yscrollcommand=scroll.set, height=12)
        self.select.bind('<Double-Button-1>', None)
        scroll.config (command=self.select.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.select.pack(side=tk.LEFT,  fill=tk.BOTH, expand=1)

        frame4 = tk.Frame(data_tab)
        frame4.pack()



    def _update(self):
        pass

    def load_data(self):
        askload = messagebox.askquestion("Load file", "Are you sure to load a database from file? All unsaved data will be lost.", icon='warning')
        if askload == 'yes':
            pass

    def clear_data(self):
        self.data = pd.DataFrame()
        self.currfile = None

    def new_db(self):
        askload = messagebox.askquestion("New file", "Are you sure to open a new database? All unsaved data will be lost.", icon='warning')
        if askload == 'yes':
            pass

    def saveas_db(self):
        self.currfile = filedialog.asksaveasfilename(title='Save data as', initialdir=self.dir)
        self.save_db()

    def save_data(self):
        if self.currfile is None:
            self.saveas_db()
        with io.open(self.currfile, 'w+', encoding='utf8') as f:
            #yaml.dump(self.db, f, default_flow_style=False, allow_unicode=True, canonical=False)
            pass
        self.master.title("Backtesting.py Simulation - {}".format(self.currfile))


if __name__ == '__main__':
    root = tk.Tk()
    app = BacktestWindow(root)
    root.mainloop()








### Gooey Examples

from gooey import Gooey, GooeyParser


@Gooey()
def main():
    parser = GooeyParser(description='Process some integers.')

    parser.add_argument(
        'required_field',
        metavar='Some Field',
        help='Enter some text!')

    parser.add_argument(
        '-f', '--foo',
        metavar='Some Flag',
        action='store_true',
        help='I turn things on and off')

    parser.parse_args()
    print('Hooray!')
"""
@Gooey
def GooeyDialog():
    parser = GooeyParser(description="Backtesting on Written Algorithms")

    parser.add_argument('--verbose', help='be verbose', dest='verbose',
                        action='store_true', default=False)

    subs = parser.add_subparsers(help='commands', dest='command')

    data_parser = subs.add_parser(
        'fetch data', help='Fetch and Prepare Data from Database')
        
    sDate.add_argument("End Date", widget="DateChooser")
    sDate.add_argument("End Time", widget="TimeChooser")

    curl_parser.add_argument('Path',
                             help='URL to the remote server',
                             type=str, widget='FileChooser')
    curl_parser.add_argument('--connect-timeout',
                             help='Maximum time in seconds that you allow curl\'s connection to take')
    curl_parser.add_argument('--user-agent',
                             help='Specify the User-Agent string ')
    curl_parser.add_argument('--cookie',
                             help='Pass the data to the HTTP server as a cookie')
    curl_parser.add_argument('--dump-header', type=argparse.FileType(),
                             help='Write the protocol headers to the specified file')
    curl_parser.add_argument('--progress-bar', action="store_true",
                             help='Make curl display progress as a simple progress bar')
    curl_parser.add_argument('--http2', action="store_true",
                             help='Tells curl to issue its requests using HTTP 2')
    curl_parser.add_argument('--ipv4', action="store_true",
                             help=' resolve names to IPv4 addresses only')

    # ########################################################
    siege_parser = subs.add_parser(
        'siege', help='Siege is an http/https regression testing and benchmarking utility')
    siege_parser.add_argument('--get',
                              help='Pull down headers from the server and display HTTP transaction',
                              type=str)
    siege_parser.add_argument('--concurrent',
                              help='Stress the web server with NUM number of simulated users',
                              type=int)
    siege_parser.add_argument('--time',
                              help='allows you to run the test for a selected period of time',
                              type=int)
    siege_parser.add_argument('--delay',
                              help='simulated user is delayed for a random number of seconds between one and NUM',
                              type=int)
    siege_parser.add_argument('--message',
                              help='mark the log file with a separator',
                              type=int)

    # ########################################################
    ffmpeg_parser = subs.add_parser(
        'ffmpeg', help='A complete, cross-platform solution to record, convert and stream audio and video')
    ffmpeg_parser.add_argument('Output',
                               help='Pull down headers from the server and display HTTP transaction',
                               widget='FileSaver', type=argparse.FileType())
    ffmpeg_parser.add_argument('--bitrate',
                               help='set the video bitrate in kbit/s (default = 200 kb/s)',
                               type=str)
    ffmpeg_parser.add_argument('--fps',
                               help='set frame rate (default = 25)',
                               type=str)
    ffmpeg_parser.add_argument('--size',
                               help='set frame size. The format is WxH (default 160x128)',
                               type=str)
    ffmpeg_parser.add_argument('--aspect',
                               help='set aspect ratio (4:3, 16:9 or 1.3333, 1.7777)',
                               type=str)
    ffmpeg_parser.add_argument('--tolerance',
                               help='set video bitrate tolerance (in kbit/s)',
                               type=str)
    ffmpeg_parser.add_argument('--maxrate',
                               help='set min video bitrate tolerance (in kbit/s)',
                               type=str)
    ffmpeg_parser.add_argument('--bufsize',
                               help='set ratecontrol buffere size (in kbit)',
                               type=str)
    sDate = parser.add_argument_group()
    sDate.add_argument(
        '--startdate',
        metavar='Start Date',
        widget="DateChooser"
    )
    sDate.add_argument(
        '--starttime',
        metavar='Start Time',
        widget='TimeChooser'
    )
    # sDate.add_argument(
    #     '--starttime',
    #     metavar='Start Time',
    #     help='Load a Previous save file',
    #     dest='filename',
    #     widget='Dropdown',
    #     choices=list_savefiles(),
    #     gooey_options={
    #         'validator': {
    #             'test': 'user_input != "Select Option"',
    #             'message': 'Choose a save file from the list'
    #         }
    #     }
    # )
    
"""