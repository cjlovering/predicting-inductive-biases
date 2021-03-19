import os

import setup
import tables
import scatterplots
import lineplots

if __name__ == "__main__":

    if not os.path.exists("./figures"):
        os.mkdir("./figures")

    if not os.path.exists("./files"):
        os.mkdir("./files")

    setup.main()
    tables.main()
    scatterplots.main()
    lineplots.main()
