import pandas as pd
import numpy as np
import xlsxwriter
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
BELGIAN_BASE_ID = 1000
N_PARTICIPANTS = 39
DATA_DIR = 'data'
FILENAMES = [f'individual_upv_{i}.xlsx' for i in range(BELGIAN_BASE_ID + 1, BELGIAN_BASE_ID + N_PARTICIPANTS + 1)]
TAB_NAMES = ['t01  Test_gestures', 't02  Test_mouse']

# Initialize 3D structure to hold cleaned data: [Participant][Variable][Values]
# Variables 0-3 are Gestures, 4-7 are Mouse
eeg_cleaned_data = [[[] for _ in range(8)] for _ in range(N_PARTICIPANTS)]

print("--- Starting Data Loading & Pre-processing ---")

# ==========================================
# 1. DATA LOADING & FILTERING
# ==========================================
for idpart, filename in enumerate(FILENAMES):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Warning: File {filename} not found. Skipping.")
        continue

    print(f"Processing Participant {idpart + 1}/{N_PARTICIPANTS}", end='\r', flush=True)
    
    idvar_offset = 0 # 0 for gestures, 4 for mouse
    
    for tab_name in TAB_NAMES:
        try:
            # Header=2 implies the 3rd row contains the actual headers
            df = pd.read_excel(file_path, sheet_name=tab_name, header=2)
            
            # Filter specific interaction periods and quality
            if 'CatalogInteraction' in df.columns:
                df = df.loc[df['CatalogInteraction'] == 1]
            if 'quality' in df.columns:
                df = df.loc[df['quality'] == 1]

            # Select only relevant metric columns (ignoring time and markers)
            # Based on Appendix 5 logic: 
            # We expect columns like: 'engagement', 'memorization', 'valence', 'workload'
            # The code specifically excludes 'time', 'Tutorial', 'Video', 'User marker', 'x', 'y', 'click'
            
            cols_to_keep = []
            for col in df.columns:
                c_lower = col.lower()
                if not any(x in c_lower for x in ['time', 'quality', 'tutorial', 'video', 'catalog', 'user', 'x', 'y', 'click']):
                    cols_to_keep.append(col)
            
            df = df[cols_to_keep]
            
            # Ensure we have exactly 4 columns (Engagement, Memorization, Valence, Workload)
            # If names differ in raw files, standardizing column names is recommended here.
            # Assuming file structure matches thesis description.
            
            # Clean non-integer values
            for i, column in enumerate(df.columns):
                # Ensure we don't exceed the 4 variables per condition
                if i >= 4: break 
                
                clean_vals = []
                for value in df[column]:
                    try:
                        clean_vals.append(float(value))
                    except:
                        continue
                
                eeg_cleaned_data[idpart][idvar_offset + i] = np.array(clean_vals)
            
            idvar_offset += 4 # Move to mouse variables for next iteration
            
        except Exception as e:
            print(f"\nError processing {filename} sheet {tab_name}: {e}")

print("\nData Loading Completed.")

# ==========================================
# 2. OUTLIER MANAGEMENT (IQR Filter)
# ==========================================
def outliers_change(eeg_data):
    """
    Applies Interquartile Range (IQR) filter.
    Uses Multiplier = 3 (Outer Boundary) as specified in Appendix 5.
    Imputes outliers with the boundary value.
    """
    eeg_filtered = [[[] for _ in range(8)] for _ in range(N_PARTICIPANTS)]
    
    # Process variables 0-3 (Gesture) and compare with 4-7 (Mouse)
    for idvar in range(4):
        for idpart in range(N_PARTICIPANTS):
            # Combine data to calculate common IQR for the participant
            combined_data = np.concatenate((eeg_data[idpart][idvar], eeg_data[idpart][idvar+4]), axis=None)
            
            if len(combined_data) == 0: continue

            Q1 = np.percentile(combined_data, 25)
            Q3 = np.percentile(combined_data, 75)
            IQR = Q3 - Q1
            range_inf = Q1 - (3 * IQR)
            range_sup = Q3 + (3 * IQR)

            # Filter Gesture (idvar)
            cleaned_gest = []
            for val in eeg_data[idpart][idvar]:
                if val > range_sup: cleaned_gest.append(range_sup)
                elif val < range_inf: cleaned_gest.append(range_inf)
                else: cleaned_gest.append(val)
            eeg_filtered[idpart][idvar] = np.array(cleaned_gest)

            # Filter Mouse (idvar + 4)
            cleaned_mouse = []
            for val in eeg_data[idpart][idvar+4]:
                if val > range_sup: cleaned_mouse.append(range_sup)
                elif val < range_inf: cleaned_mouse.append(range_inf)
                else: cleaned_mouse.append(val)
            eeg_filtered[idpart][idvar+4] = np.array(cleaned_mouse)
            
    return eeg_filtered

print("--- Handling Outliers ---")
eeg_filtered_data = outliers_change(eeg_cleaned_data)

# ==========================================
# 3. FEATURE EXTRACTION & METRICS
# ==========================================

# A. Proportion of High Activity (> 100)
print("--- Computing Proportions > 100 ---")
proportion_above_hundred = np.zeros((N_PARTICIPANTS, 8))
wb_prop = xlsxwriter.Workbook('%above100.xlsx')
ws_prop = wb_prop.add_worksheet()
headers = ['ID', 'EngaGest', 'MemoGest', 'ValeGest', 'WorkGest', 'EngaMous', 'MemoMous', 'ValeMous', 'WorkMous']
ws_prop.write_row(0, 0, headers)

for idpart in range(N_PARTICIPANTS):
    ws_prop.write(idpart + 1, 0, BELGIAN_BASE_ID + idpart + 1)
    for idvar in range(8):
        data = eeg_filtered_data[idpart][idvar]
        if len(data) > 0:
            count = np.sum(data > 100)
            prop = count / len(data)
            proportion_above_hundred[idpart][idvar] = prop
            ws_prop.write(idpart + 1, idvar + 1, prop)
wb_prop.close()

# B. Mean Values
print("--- Computing Means ---")
wb_mean = xlsxwriter.Workbook('mean.xlsx')
ws_mean = wb_mean.add_worksheet()
ws_mean.write_row(0, 0, headers)

for idpart in range(N_PARTICIPANTS):
    ws_mean.write(idpart + 1, 0, BELGIAN_BASE_ID + idpart + 1)
    for idvar in range(8):
        data = eeg_filtered_data[idpart][idvar]
        if len(data) > 0:
            mean_val = np.mean(data)
            ws_mean.write(idpart + 1, idvar + 1, mean_val)
wb_mean.close()

# C. Duration Correction & Average Max Peak
print("--- Computing Corrected Max Peaks (Simulation) ---")

def duration_correction(eeg_data):
    """
    Randomly removes data points from the longer session to match 
    the duration of the shorter session.
    """
    corrected_data = [[np.copy(eeg_data[p][v]) for v in range(8)] for p in range(N_PARTICIPANTS)]
    
    for idpart in range(N_PARTICIPANTS):
        for idvar in range(4): # Loop through variable pairs
            len_gest = len(corrected_data[idpart][idvar])
            len_mouse = len(corrected_data[idpart][idvar+4])
            
            if len_gest > len_mouse:
                diff = len_gest - len_mouse
                if len_gest > 1:
                    # Randomly delete indices
                    indices = np.random.choice(len_gest, diff, replace=False)
                    corrected_data[idpart][idvar] = np.delete(corrected_data[idpart][idvar], indices)
            elif len_mouse > len_gest:
                diff = len_mouse - len_gest
                if len_mouse > 1:
                    indices = np.random.choice(len_mouse, diff, replace=False)
                    corrected_data[idpart][idvar+4] = np.delete(corrected_data[idpart][idvar+4], indices)
                    
    return corrected_data

SIMULATIONS_PEAK = 2000
average_max_peak = np.zeros((N_PARTICIPANTS, 8))

for i in range(SIMULATIONS_PEAK):
    print(f"Simulation {i+1}/{SIMULATIONS_PEAK}", end='\r')
    temp_data = duration_correction(eeg_filtered_data)
    
    for idpart in range(N_PARTICIPANTS):
        for idvar in range(8):
            if len(temp_data[idpart][idvar]) > 0:
                average_max_peak[idpart][idvar] += np.max(temp_data[idpart][idvar])

# Average out the peaks
average_max_peak /= SIMULATIONS_PEAK

wb_peak = xlsxwriter.Workbook('AverageCorrectedMaxPeak.xlsx')
ws_peak = wb_peak.add_worksheet()
ws_peak.write_row(0, 0, headers)
for idpart in range(N_PARTICIPANTS):
    ws_peak.write(idpart + 1, 0, BELGIAN_BASE_ID + idpart + 1)
    for idvar in range(8):
        ws_peak.write(idpart + 1, idvar + 1, average_max_peak[idpart][idvar])
wb_peak.close()
print("\nPeak Analysis Completed.")

# ==========================================
# 4. MACHINE LEARNING (Random Forest)
# ==========================================
print("--- Starting Random Forest Classification ---")

wb_ml = xlsxwriter.Workbook('DiscriminantAnalysis.xlsx')
ws_ml = wb_ml.add_worksheet()
ml_headers = ['Metric', 'Value']
ws_ml.write_row(0, 0, ml_headers)

SIMULATIONS_ML = 1000 # As specified in Appendix 5 code
metrics = {'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
feature_importance_acc = np.zeros(8) # 4 vars * 2 features (Peak + Prop)

for i in range(SIMULATIONS_ML):
    print(f"ML Simulation {i+1}/{SIMULATIONS_ML}", end='\r')
    
    # 1. Prepare Data
    # Features for Gesture: [MaxPeak, Proportion] for variable indices 0-3
    # Note: Logic assumes indices 0=Engagement, 1=Memorization, 2=Valence, 3=Workload
    # We construct rows. Each participant contributes 2 rows (1 gesture, 1 mouse)
    
    X_list = []
    y_list = []
    
    for idpart in range(N_PARTICIPANTS):
        # Gesture Row (Label 1)
        # Features: MaxPeak(Eng, Mem, Val, Work) + Prop(Eng, Mem, Val, Work)
        feats_g = []
        for v in range(4): feats_g.append(average_max_peak[idpart][v])
        for v in range(4): feats_g.append(proportion_above_hundred[idpart][v])
        X_list.append(feats_g)
        y_list.append(1)
        
        # Mouse Row (Label 0)
        # Features: MaxPeak + Prop (indices 4-7)
        feats_m = []
        for v in range(4, 8): feats_m.append(average_max_peak[idpart][v])
        for v in range(4, 8): feats_m.append(proportion_above_hundred[idpart][v])
        X_list.append(feats_m)
        y_list.append(0)

    X = np.array(X_list)
    y = np.array(y_list)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 3. Train
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # 4. Evaluate
    metrics['acc'] += accuracy_score(y_test, y_pred)
    metrics['prec'] += precision_score(y_test, y_pred, zero_division=0)
    metrics['rec'] += recall_score(y_test, y_pred, zero_division=0)
    metrics['f1'] += f1_score(y_test, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    metrics['tn'] += tn
    metrics['fp'] += fp
    metrics['fn'] += fn
    metrics['tp'] += tp
    
    feature_importance_acc += clf.feature_importances_

# Averaging results
for k in metrics:
    metrics[k] /= SIMULATIONS_ML
feature_importance_acc /= SIMULATIONS_ML

print("\n--- ML Results ---")
print(f"Accuracy: {metrics['acc']:.4f}")
print(f"Precision: {metrics['prec']:.4f}")
print(f"Recall: {metrics['rec']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")

# Save to Excel
ws_ml.write(1, 0, 'Accuracy')
ws_ml.write(1, 1, metrics['acc'])
ws_ml.write(2, 0, 'Precision')
ws_ml.write(2, 1, metrics['prec'])
ws_ml.write(3, 0, 'Recall')
ws_ml.write(3, 1, metrics['rec'])
ws_ml.write(4, 0, 'F1-Score')
ws_ml.write(4, 1, metrics['f1'])

ws_ml.write(6, 0, 'Confusion Matrix')
total_cm = metrics['tn'] + metrics['fp'] + metrics['fn'] + metrics['tp']
ws_ml.write(7, 0, 'True Negative'); ws_ml.write(7, 1, metrics['tn']/total_cm)
ws_ml.write(8, 0, 'False Positive'); ws_ml.write(8, 1, metrics['fp']/total_cm)
ws_ml.write(9, 0, 'False Negative'); ws_ml.write(9, 1, metrics['fn']/total_cm)
ws_ml.write(10, 0, 'True Positive'); ws_ml.write(10, 1, metrics['tp']/total_cm)

# Feature Importance
feature_names = [
    'MaxPeak_Eng', 'MaxPeak_Mem', 'MaxPeak_Val', 'MaxPeak_Work',
    'Prop_Eng', 'Prop_Mem', 'Prop_Val', 'Prop_Work'
]

ws_ml.write(12, 0, 'Feature Importances')
for i, name in enumerate(feature_names):
    ws_ml.write(13+i, 0, name)
    ws_ml.write(13+i, 1, feature_importance_acc[i])

wb_ml.close()
print("All tasks completed. Results saved to Excel files.")
