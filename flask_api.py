#https://startbootstrap.com/themes/sb-admin-2/
#https://stackoverflow.com/questions/45528007/flask-href-link-to-html-not-working/45528063
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import json
import pygal
from pygal.style import Style
from pygal.style import Style
client = MongoClient('localhost', 27017)
db = client['indeed2']
app = Flask(__name__)



custom_style = Style(

  legend_font_size = 30,
  background='transparent',
  plot_background='transparent',
  foreground='#53E89B',
  foreground_strong='#53A0E8',
  foreground_subtle='#630C0D',
  opacity='1',
  opacity_hover='.4',
  transition='400ms ease-in',
  colors=('#4e73df', '#36b9cc', '#1cc88a', '#f6c23e', '#e74a3b'),
  tooltip_font_size =30)


def graphgal():
    pie_chart = pygal.Pie(style=custom_style)
    pie_chart.add('IE', 19.5)
    pie_chart.add('Firefox', 36.6)
    pie_chart.add('Chrome', 36.3)
    pie_chart.add('Safari', 4.5)
    pie_chart.add('Opera', 2.3)
    pie = pie_chart.render_data_uri()
    return pie

def piegal():
    pie_chart = pygal.Pie(style=custom_style, legend_box_size=30)
    pie_chart.add('IE', 19.5)
    pie_chart.add('Firefox', 36.6)
    pie_chart.add('Chrome', 36.3)
    pie = pie_chart.render_data_uri()
    return pie
    #<input class="btn btn-primary" type="submit" name="submit_button" value="Indeed database">

@app.route('/')
@app.route('/<poste>.<ville>.<contrat>.<date_debut>.<date_fin>')
def homepage(poste = None, ville = None, contrat = None, date_debut = None, date_fin = None):
    if poste != None :
        return render_template('index.html', pie_chart = piegal(), graph_chart =  graphgal())
    return render_template('index.html')

@app.route('/mongodb')
def datatest():
    cursor = db.indeed2.find({})
    cursor = {i : dic for i,dic in enumerate(cursor)}  
    return render_template('tables-test.html', questions= cursor , len = len(cursor))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.route('/test')
def anime():
    return render_template('utilities-animation.html')
if __name__ == "__main__":
	app.run(debug = True, host='0.0.0.0', port=8080, passthrough_errors=True)