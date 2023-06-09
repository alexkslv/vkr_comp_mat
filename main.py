import flask
from keras.models import load_model

comp_mat_model = load_model('net_model_comp_mat')

app = flask.Flask(__name__, template_folder='templates')


@app.route("/", methods=['POST', 'GET'])
@app.route("/index", methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        kol_otverd = float(flask.request.form['kol_otverd'])
        epoxid     = float(flask.request.form['epoxid'])
        smola      = float(flask.request.form['smola'])
        ugol_nash  = float(flask.request.form['ugol_nash'])
        shag_nash  = float(flask.request.form['shag_nash'])
        plot_nash  = float(flask.request.form['plot_nash'])
        X = [[kol_otverd, epoxid, smola, ugol_nash, shag_nash, plot_nash]]
        result = comp_mat_model.predict(X)
        return flask.render_template('main.html', result=result,
                                              kol_otverd=kol_otverd,
                                                  epoxid=epoxid,
                                                   smola=smola,
                                               ugol_nash=ugol_nash,
                                               shag_nash=shag_nash,
                                               plot_nash=plot_nash)

if __name__ == '__main__':
    app.run()


