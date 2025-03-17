import pandas as pd, numpy as np, sqlite3, random, pickle, logging, os
import flax.linen as nn, jax.numpy as jnp, jax, optax
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

conn=sqlite3.connect('causal_analysis.db')
cursor=conn.cursor()
cursor.executescript('''
CREATE TABLE IF NOT EXISTS models(
dataset TEXT, output TEXT, inputs TEXT, model_type TEXT, cv_loss REAL,
val_loss REAL, params INTEGER, model_size REAL, combined_score REAL, blob BLOB);
CREATE TABLE IF NOT EXISTS dag_edges(
dataset TEXT, source TEXT, target TEXT, confidence REAL);
''')

random.seed(42); np.random.seed(42)

# Transformer with positional encoding
class PositionalEncoding(nn.Module):
    dim:int
    @nn.compact
    def __call__(self,x):
        pos=jnp.arange(x.shape[1])[:,None]
        i=jnp.arange(self.dim)[None,:]
        angle_rates=1/jnp.power(10000,2*(i//2)/self.dim)
        angles=pos*angle_rates
        pos_encoding=jnp.where(i%2==0,jnp.sin(angles),jnp.cos(angles))
        return x+pos_encoding

class TransformerModel(nn.Module):
    dim:int; layers:int; heads:int
    @nn.compact
    def __call__(self,x):
        x=nn.Dense(self.dim)(x)
        x=PositionalEncoding(self.dim)(x)
        for _ in range(self.layers):
            attn=nn.SelfAttention(num_heads=self.heads,qkv_features=self.dim)(x)
            x=nn.LayerNorm()(x+attn)
            dense=nn.Dense(self.dim*4)(x)
            dense=nn.relu(dense)
            dense=nn.Dense(self.dim)(dense)
            x=nn.LayerNorm()(x+dense)
        return nn.Dense(1)(x.mean(axis=1)).squeeze()

def scale_params(n_samples):
    base=min(max(int((n_samples/1e5)**0.5),1),8)
    return dict(dim=base*32,layers=base,heads=min(base,8))

def train_transformer(X,y,params,epochs=50):
    key=jax.random.PRNGKey(0)
    model=TransformerModel(**params)
    opt=optax.adam(1e-3)
    state=model.init(key,X)
    opt_state=opt.init(state)

    @jax.jit
    def loss_fn(params,X,y): return jnp.mean((model.apply(params,X)-y)**2)

    @jax.jit
    def update(params,opt_state,X,y):
        grads=jax.grad(loss_fn)(params,X,y)
        updates,opt_state=opt.update(grads,opt_state)
        return optax.apply_updates(params,updates),opt_state

    for _ in range(epochs):
        state,opt_state=update(state,opt_state,X,y)
    return state,loss_fn(state,X,y).item()

def preprocess(df):
    df=df.dropna(axis=1,thresh=int(0.7*len(df))).dropna()
    num=df.select_dtypes(np.number).columns
    cat=df.select_dtypes(['object','bool','category']).columns
    if num.any():df[num]=StandardScaler().fit_transform(df[num])
    if cat.any():
        enc=OneHotEncoder(drop='first',sparse=False).fit(df[cat])
        df=pd.concat([df[num],pd.DataFrame(enc.transform(df[cat]))],axis=1)
    return df.reset_index(drop=True)

def combined_score(cv_loss,val_loss,params,model_size):
    return (cv_loss+val_loss)*np.log1p(params)*np.log1p(model_size)

def model_size(blob): return len(blob)/(1024*1024)

def infer_dag(dataset,results):
    edges={}
    df=pd.DataFrame(results,columns=['output','inputs','score'])
    for output,inputs,score in results:
        threshold=np.percentile(df[df.output==output].score,2)
        if score<=threshold:
            for src in inputs:edges[(src,output)]=edges.get((src,output),0)+1
    total=sum(edges.values())
    for (src,tgt),cnt in edges.items():
        cursor.execute('INSERT INTO dag_edges VALUES(?,?,?,?)',
                       (dataset,src,tgt,cnt/total))
    conn.commit()

def causal_pipeline(df,name,num_trials=200,use_transformer=True):
    cols=df.columns
    train_df,val_df=train_test_split(df,test_size=0.2,shuffle='time' not in name.lower())
    results=[]

    for output in cols:
        inputs=[c for c in cols if c!=output]
        subsets=set()
        while len(subsets)<num_trials:
            subset=tuple(sorted(random.sample(inputs,random.randint(1,len(inputs)))))
            if subset in subsets:continue
            subsets.add(subset)

            Xtr,ytr=train_df[list(subset)].values,train_df[output].values
            Xval,yval=val_df[list(subset)].values,val_df[output].values

            if use_transformer:
                params=scale_params(len(Xtr))
                Xtr_exp,jXtr=jnp.expand_dims(Xtr,1),jnp.array(Xtr)
                model,state_loss=train_transformer(Xtr_exp,ytr,params)
                val_pred=TransformerModel(**params).apply(model,jnp.expand_dims(Xval,1))
                cv_loss,val_loss=state_loss,mean_squared_error(yval,np.array(val_pred))
                blob=pickle.dumps(model)
                param_count=sum([np.prod(p.shape) for p in jax.tree_util.tree_leaves(model)])
                m_type='transformer'
            else:
                xmodel=xgb.XGBRegressor(n_jobs=-1,tree_method='hist').fit(Xtr,ytr)
                cv_loss=-cross_val_score(xmodel,Xtr,ytr,cv=5,scoring='neg_mean_squared_error').mean()
                val_loss=mean_squared_error(yval,xmodel.predict(Xval))
                blob=pickle.dumps(xmodel)
                param_count=xmodel.get_booster().trees_to_dataframe().shape[0]
                m_type='xgboost'

            size=model_size(blob)
            c_score=combined_score(cv_loss,val_loss,param_count,size)
            cursor.execute('INSERT INTO models VALUES(?,?,?,?,?,?,?,?,?,?)',
                           (name,output,','.join(subset),m_type,cv_loss,val_loss,param_count,size,c_score,blob))
            results.append((output,subset,c_score))
        conn.commit()
        logging.info(f"Completed {output} in {name}")
    infer_dag(name,results)

def main():
    datasets={'US_Accidents':'us_accidents.csv',
              'BeijingAirQuality':'beijing_air.csv',
              'RossmannSales':'rossmann_sales.csv'}
    for name,path in datasets.items():
        df=preprocess(pd.read_csv(path))
        causal_pipeline(df,name,use_transformer=True) # Set False for XGBoost
    logging.info("All datasets processed.")

if __name__=='__main__':
    main()
