import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import graph_construct

graph_root = 'data/graph'
graph_name = 'population_graph_all_structural_euler_no_edges_sex.pt'
population_graph = graph_construct.load_population_graph(graph_root, graph_name)

X_train = population_graph.x[population_graph.train_mask].cpu().detach().numpy()
X_validate = population_graph.x[population_graph.validate_mask].cpu().detach().numpy()

y_train = population_graph.y[population_graph.train_mask].cpu().detach().numpy().flatten()
y_validate = population_graph.y[population_graph.validate_mask].cpu().detach().numpy().flatten()

xgb_model = xgb.XGBRegressor(learning_rate=0.01, n_jobs=-1, verbosity=2)

xgb_model.fit(X_train, y_train, verbose=True)
print('pearsonr', pearsonr(y_validate, xgb_model.predict(X_validate)))
print('r2_score', r2_score(y_validate, xgb_model.predict(X_validate)))
