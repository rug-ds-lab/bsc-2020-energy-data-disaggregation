def class_to_latex(data: {}, labels: []):
    string = "\\begin{table}[]\n\\centering\n\\begin{tabular}{||c | c | c | c | c ||}\n"
    string += "\\hline\nclassification report &  precision & recall & f1-score & support  \\\\ [0.5ex]\n"
    string += "\\hline\n"
    for key, label in enumerate(labels):
        if str(key) in data.keys():
            string += label.replace("_", " ") + " & " + "%.2f" % data[str(key)]['precision'] + " & " + "%.2f" % \
                      data[str(key)]['recall'] + " & " + "%.2f" % data[str(key)]['f1-score'] + " & " + \
                      str(data[str(key)]['support']) + "\\\\\\hline\n"
    string += "\\end{tabular}\\end{table}"
    return string


def conf_to_latex(data: [[]], labels: []):
    string = "\\begin{table}[]\n\\centering\n\\begin{tabular}{||" + " c |" * (len(labels) + 1) + "|}\n"
    string += "\\hline\n confusion matrix"
    for label in labels:
        string += " & " + label.replace("_", " ")
    string += "\\\\ [0.5ex]\n\\hline\n"

    for i, label in enumerate(labels):
        string += label.replace("_", " ")
        for d in data[i]:
            string += " & " + str(d)
        string += "\\\\\\hline\n"

    string += "\\end{tabular}\\end{table}"
    return string
