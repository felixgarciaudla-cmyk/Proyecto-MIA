# Análisis complementarios reportados en la sección 7.3 de la tesis (ejecutar en Colab con los CSV limpios y el modelo ganador)
# !pip install shap
import joblib, time, numpy as np, pandas as pd, shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, recall_score, roc_auc_score, log_loss

tr = pd.read_csv('2_train_mora_limpio.csv', dtype={'ID': 'string'})
va = pd.read_csv('4_validation_mora_limpio.csv', dtype={'ID': 'string'})
te = pd.read_csv('3_test_mora_limpio.csv', dtype={'ID': 'string'})
m = joblib.load('modelo_ganador_mora.joblib'); pre, mod = m['preprocesador'], m['modelo']
Xte = pre.transform(te[m['variables']])

# 1. Métricas adicionales en test (ROC-AUC macro, log loss) y latencia
p = mod.predict_proba(Xte)
print('ROC-AUC macro OvR:', roc_auc_score(te.mora, p, multi_class='ovr', average='macro'), '| Log loss:', log_loss(te.mora, p, labels=mod.classes_))
t = time.time()
for i in range(200): mod.predict_proba(pre.transform(te[m['variables']].iloc[[i]]))
print('Latencia media por registro (ms):', (time.time() - t) / 200 * 1000)

# 2. SHAP global (Figura 8)
names = pre.get_feature_names_out()
Xs = pd.DataFrame(Xte, columns=names).sample(800, random_state=42)
sv = shap.TreeExplainer(mod).shap_values(Xs)
orig = [next(v for v in sorted(m['variables'], key=len, reverse=True) if n.split('__', 1)[1] == v or n.split('__', 1)[1].startswith(v + '_')) for n in names]
imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=orig, columns=mod.classes_).groupby(level=0).sum()
print(imp.sum(axis=1).sort_values(ascending=False).head(10))

# 3. Análisis de sensibilidad (Tabla 9)
base = ['ID', 'fecha_reporte', 'atraso', 'mora']
escenarios = {'A': [], 'B': ['clasif_sbs'], 'C': ['clasif_sbs', 'mora_max_3m'], 'D': ['clasif_sbs', 'mora_max_3m', 'mora_max_6m', 'mora_max_12m']}
for n, ex in escenarios.items():
    v = [c for c in tr.columns if c not in base + ex]
    num = tr[v].select_dtypes('number').columns.tolist(); cat = [c for c in v if c not in num]
    pr = ColumnTransformer([('n', StandardScaler(), num), ('c', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)], sparse_threshold=0).fit(tr[v])
    hgb = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=31, learning_rate=0.08, l2_regularization=1.0, early_stopping=True, random_state=42)
    hgb.fit(pr.transform(tr[v]), tr.mora, sample_weight=compute_sample_weight('balanced', tr.mora))
    lr = LogisticRegression(max_iter=2500, class_weight='balanced', random_state=42).fit(pr.transform(tr[v]), tr.mora)
    print(n, len(v), 'HGB val', round(f1_score(va.mora, hgb.predict(pr.transform(va[v])), average='macro'), 3),
          'HGB test', round(f1_score(te.mora, hgb.predict(pr.transform(te[v])), average='macro'), 3),
          'LR test', round(f1_score(te.mora, lr.predict(pr.transform(te[v])), average='macro'), 3))

# 4. Auditoría de equidad (Tabla 10)
te['pred'] = mod.predict(Xte); te['y_b'] = te.mora != 'AL DÍA'; te['p_b'] = te.pred != 'AL DÍA'
te['edad_g'] = pd.cut(te.edad, [17, 29, 44, 59, 100], labels=['18-29', '30-44', '45-59', '60+'])
for g in ['zona', 'nivel_educ', 'edad_g']:
    r = te.groupby(g, observed=True).apply(lambda d: pd.Series({'n': len(d), 'f1_macro': f1_score(d.mora, d.pred, average='macro'),
        'TVP': recall_score(d.y_b, d.p_b), 'TFP': ((~d.y_b) & d.p_b).sum() / max((~d.y_b).sum(), 1)}))
    print(r.round(3)); print('Brecha TVP:', round(r.TVP.max() - r.TVP.min(), 3))
