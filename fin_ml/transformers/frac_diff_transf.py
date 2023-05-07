import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller
from sklearn import set_config
set_config(transform_output="pandas")

class FracDiff(BaseEstimator, TransformerMixin):

    def __init__(self,
                 d: float = 1., 
                 method: str = 'fww', 
                 thres: float = 1e-2,
                 minimum: bool = False,
                 min_dict: dict = {'interv': [0., 1.], 
                                   'step': 1e-1, 
                                   'c_val_idx': int(1)}):
            
        self.d = d 
        self.method = method
        self.thres = thres
        self.minimum = minimum
        self.min_dict = min_dict

    def get_weights(self,
                    d: float = 1.,
                    size: int = 1,
                    method: str = 'fww',
                    thres: float = 1.e-2) -> np.array:
        '''    
        Calcula os pesos a serem aplicados as defasagens da série.
        Adaptado de:
        https://www.ostirion.net/post/stock-price-fractional-differentiation-best-fraction-finder
        e do cap. 5 do livro Advances in Financial Machine Learning - Marcos M. Lopez de Prado - Ed. 1ª 
        
        Args:        
        -----            
        d: float 
            Ordem de diferenciação.        
          
        thres: float
            Nível de tolerância o para definir quais pesos serão descartados.
            
        Returns: 
        -----
        w: Numpy float array       
            Vetor de tamanho k com os pesos w.
        '''
        
        w = [1.0]
        k = 1
 
        if method == 'fww':                                      # selecionando o método
            
            while True:
                w_ = -w[-1]/k*(d-k+1)                            # calcula o peso até a ordem k, enquanto menor que o nível de tolerância
                if abs(w_) <= thres:
                    break
                w.append(w_)
                k += 1
                
        elif method == 'std':
            
            for k in range(k, size):
                w_ = -w[-1]/k*(d-k+1)                           # calcula o peso considerando toda a amostra de T até T-k
                w.append(w_)
    
        w = np.array(w[::-1]).reshape(-1, 1)
        
        return w
    
    def calc_frac_diff(self,
                       X: pd.Series(dtype = float), 
                       d: float = 1., 
                       method: str = 'fww', 
                       thres: float = 1e-2        ) -> pd.Series(dtype = float):
        '''    
        Calcula a série diferenciada francionalmente.
        Adaptado de:
        https://www.ostirion.net/post/stock-price-fractional-differentiation-best-fraction-finder
        e do cap. 5 do livro Advances in Financial Machine Learning - Marcos M. Lopez de Prado - Ed. 1ª 
        
        Args:        
        -----
        X: Pandas Series float
            Série a ser diferenciada.        
                
        d: float 
            Ordem de diferenciação.        
          
        method: string 
            Método a ser utilizado. Quando 'fww', aplica método da
            janela de comprimento fixo, quando 'std', aplica o 
            método janela expansiva.
            
        threshold: float
            Nível de tolerância para definir quais pesos serão descartados. 
            
        Returns: 
        -----
        X: Pandas Série float      
            Série diferenciada fracionalmente.
        '''
        
        l = 1
        size = X.shape[0]
        
        if method == 'fww':
            w = self.get_weights(d, size, method, thres)     # calcula os pesos a serem aplicados na série
            l = len(w)
            
            if l > size:
                method = 'std'
    
        elif method == 'std':
            w = self.get_weights(d, size, method, thres)     # calcula os pesos a serem aplicados na série
            w_ = np.cumsum(abs(w))             
            w_ /= w_[-1]
            l = w_[w_>thres].shape[0]                   # elimina perda-ponderada relativa maior que um nível de tolerância
          
        results = {}
        r = range(l, size)
    
        for idx in r:
            
            if not np.isfinite(X.iloc[idx]): continue   # elimina os NAs
            
            if method == 'std':         
                results[idx] = np.dot(w[-(idx):].T, 
                                      X.iloc[:idx].to_numpy(dtype=float))[0]            # diferenciando a série
            
            elif method == 'fww':
                results[idx] = np.dot(w.T, 
                                      X.iloc[(idx-l):idx].to_numpy(dtype=float))[0]     # diferenciando a série
        
        X_tilda = pd.Series(results)
        X_tilda = X_tilda.reindex(X[l:].index)
    
        return X_tilda

    def min_fd_order(self,
                     X: pd.Series(dtype = float),
                     interv: float = [0., 1.], 
                     step: float = 1e-1, 
                     c_val_idx: int = 1,  
                     method: str = 'fww', 
                     thres: float = 1e-2         ) -> pd.Series(dtype=float):
        '''    
        Calcula a menor ordem d, que passe no teste ADF a um determinado nível de confiança, e
        retorna a série diferenciada fracionalmente para esta ordem, bem como esta.

        Args:        
        -----
        X: Pandas Series float
            Série a ser diferenciada.
        
        interv: float list
            Intervalo das ordens a serem usadas na diferenciação.      
                
        step: float 
            Incremento da ordem a ser usada na diferenciação.        
          
        c_val_idx: int
            Há 3 possíveis valores: 0, 1 e 2, para definir qual o nível
            de confiança do teste ADF. Neste caso 0: 1%, 1: 5%, 2: 10%.
            
        method: string
            Método a ser utilizado. Quando 'fww', aplica método da
            janela de comprimento fixo, quando 'std', aplica o 
            método janela expansiva.

        thres: float
            Nível de tolerância para definir quais pesos serão descartados.             
            
        Returns: 
        -----
        X: Pandas Série float      
            Série diferenciada fracionalmente.
        '''
        
        d = []  
        series_ = pd.Series(dtype=float) 
        d_ = interv[0]
        
        while d_ <= interv[1]:
            
            series_ = self.calc_frac_diff(X, d_, method, thres)     # diferenciando a série para cada ordem d
                                        
            adf = adfuller(series_,                                 # aplicando teste ADF para a séri diferenciada com ordem d                                             
                           maxlag = 1,
                           regression = 'c',
                           autolag = None)
            
            # critical values = {'1%': -3.4369193380671, '5%': -2.864440383452517, '10%': -2.56831430323573}
            # valores críticos aproximados do teste ADF segundo MacKinnon (1994, 2010
            if adf[0] < list(adf[4].items())[c_val_idx][1]:         # aplicando o teste
                break
                
            d_ += step
     
        X_tilda = series_
        d.append(d_)
    
        return X_tilda, d

    #Estimando o modelo
    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
               
        return self
        
    #Transformando os dados
    def transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ) -> pd.DataFrame(dtype=float):
        '''    
        Transforma a séria original para uma série diferenciada fracionalmente.
        Pode-se escolher entre métodos da janela fixa e expansiva, e também escolher
        uma ordem de diferenciação ou achar a menor ordem de diferenciação, que passe
        o teste ADF para um certo nível de confiança
        Adaptado de:

        Args:        
        -----            
        self: object
            Todas as entradas da função __init__.        

        X: Pandas DataFrame float
            Colunas do dataframe são as séries a serem diferenciadas.

        Returns: 
        -----
        X_tilda: Pandas DataFrame float     
            DataFrame com todas as colunas diferenciadas fracionalmente.
        '''
 
        if isinstance(X,pd.Series):               
            X = X.to_frame('0')                                                                 # caso seja somente um Pandas Series, transforma para um DataFrame

        X_tilda = pd.DataFrame(columns = [str(col) + '_fd' for col in list(X.columns)])         # inicializando DataFrame a ser retornado e nomeando suas colunas
    
        for col in X.columns:
            series_ = pd.Series(dtype=float)
            
            if self.minimum:           
                series_, _ = self.min_fd_order(X[col],                                          # calcula séria diferenciada a partir da menor ordem d, pelo teste ADF
                                          self.min_dict['interv'], 
                                          self.min_dict['step'], 
                                          self.min_dict['c_val_idx'],  
                                          self.method, 
                                          self.thres)

            else:                                                                                   
                series_ = self.calc_frac_diff(X[col], self.d, self.method, self.thres)          # calcula séria diferenciada a partir de uma ordem d escolhida

            X_tilda[str(col)+'_fd'] = pd.Series(series_.values, index = series_.index)          # alocando as séries transformadas ao DataFrame, garantindo integridade dos índices

        return X_tilda