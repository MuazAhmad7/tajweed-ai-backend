from datasets import load_dataset
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
print("Loading qalqalah dataset...")
qalqalah_ds = load_dataset("hetchyy/surah_ikhlas_qalqala")

print("Dataset information:")
print(qalqalah_ds)
print("\n" + "="*50)

# Check if dataset has train/test splits or is a single dataset
if 'train' in qalqalah_ds:
    dataset = qalqalah_ds['train']
    print(f"Using train split with {len(dataset)} samples")
else:
    dataset = qalqalah_ds
    print(f"Using full dataset with {len(dataset)} samples")

print("\nDataset features:")
print(dataset.features)
print("\n" + "="*50)

# Remove audio column to avoid decoding issues and convert to pandas
print("Converting to pandas (excluding audio column)...")
columns_to_keep = [col for col in dataset.column_names if col != 'audio']
df = dataset.remove_columns(['audio']).to_pandas()

print(f"\nDataFrame shape: {df.shape}")
print(f"Column names: {list(df.columns)}")

print("\nFirst few rows:")
print(df.head(10))

print("\nDataFrame info:")
print(df.info())

# Analyze qalqalah columns
print("\n" + "="*50)
print("Analyzing qalqalah columns...")

qalqalah_columns = [col for col in df.columns if 'qalqala' in col.lower()]
print(f"Qalqalah columns found: {qalqalah_columns}")

for col in qalqalah_columns:
    print(f"\nColumn '{col}':")
    print(f"  Data type: {df[col].dtype}")
    print(f"  Unique values: {df[col].unique()}")
    print(f"  Value counts:")
    print(f"    {df[col].value_counts().to_dict()}")
    
    # Count positive samples (assuming 1 = positive qalqalah)
    positive_samples = df[df[col] == 1]
    print(f"  Positive samples (value=1): {len(positive_samples)}")

# Based on Surah Al-Ikhlas analysis, identify which columns correspond to Daal
print("\n" + "="*50)
print("Surah Al-Ikhlas Analysis:")
print("Verse 1: قُلْ هُوَ اللَّهُ أَحَدٌ - 'ahad' ends with Daal")
print("Verse 2: اللَّهُ الصَّمَدُ - 'samad' ends with Daal") 
print("Verse 3: لَمْ يَلِدْ - 'yalid' ends with Daal")
print("Verse 4: وَلَمْ يُولَدْ - 'yulad' ends with Daal")

# Extract Daal qalqalah samples
print("\n" + "="*50)
print("EXTRACTING DAAL QALQALAH POSITIVE SAMPLES:")

daal_columns = ['qalqala_ahad_v1', 'qalqala_samad', 'qalqala_yalid', 'qalqala_yulad', 'qalqala_ahad_v4']
all_daal_positive = pd.DataFrame()

for col in daal_columns:
    if col in df.columns:
        positive_samples = df[df[col] == 1].copy()
        positive_samples['qalqalah_source'] = col
        positive_samples['word_ending_with_daal'] = col.replace('qalqala_', '')
        all_daal_positive = pd.concat([all_daal_positive, positive_samples], ignore_index=True)
        print(f"\n{col}: {len(positive_samples)} positive samples")

print(f"\nTotal Daal qalqalah positive samples: {len(all_daal_positive)}")

if len(all_daal_positive) > 0:
    print("\nSample of positive Daal qalqalah data:")
    print(all_daal_positive[['verse', 'qalqalah_source', 'word_ending_with_daal']].head(10))
    
    # Save the positive samples
    output_file = 'daal_qalqalah_positive_samples.csv'
    all_daal_positive.to_csv(output_file, index=False)
    print(f"\nPositive Daal qalqalah samples saved to: {output_file}")
    
    # Summary statistics
    print("\nSummary by word:")
    summary = all_daal_positive['word_ending_with_daal'].value_counts()
    print(summary)
    
    print("\nSummary by verse:")
    verse_summary = all_daal_positive['verse'].value_counts().sort_index()
    print(verse_summary)

print("\n" + "="*50)
print("EXTRACTING DAAL QALQALAH NEGATIVE SAMPLES:")

# Extract negative samples (where any Daal qalqalah column = 0)
all_daal_negative = pd.DataFrame()

for col in daal_columns:
    if col in df.columns:
        negative_samples = df[df[col] == 0].copy()
        negative_samples['qalqalah_source'] = col
        negative_samples['word_ending_with_daal'] = col.replace('qalqala_', '')
        all_daal_negative = pd.concat([all_daal_negative, negative_samples], ignore_index=True)
        print(f"\n{col}: {len(negative_samples)} negative samples")

print(f"\nTotal Daal qalqalah negative samples: {len(all_daal_negative)}")

if len(all_daal_negative) > 0:
    print("\nSample of negative Daal qalqalah data:")
    print(all_daal_negative[['verse', 'qalqalah_source', 'word_ending_with_daal']].head(10))
    
    # Save the negative samples
    negative_output_file = 'daal_qalqalah_negative_samples.csv'
    all_daal_negative.to_csv(negative_output_file, index=False)
    print(f"\nNegative Daal qalqalah samples saved to: {negative_output_file}")
    
    # Summary statistics for negative samples
    print("\nNegative samples summary by word:")
    negative_summary = all_daal_negative['word_ending_with_daal'].value_counts()
    print(negative_summary)
    
    print("\nNegative samples summary by verse:")
    negative_verse_summary = all_daal_negative['verse'].value_counts().sort_index()
    print(negative_verse_summary)

print("\n" + "="*50)
print("CREATING BALANCED DATASET FOR BINARY CLASSIFICATION:")

if len(all_daal_positive) > 0 and len(all_daal_negative) > 0:
    # Create balanced dataset
    min_samples = min(len(all_daal_positive), len(all_daal_negative))
    
    # Sample equal numbers from positive and negative
    balanced_positive = all_daal_positive.sample(n=min_samples, random_state=42)
    balanced_negative = all_daal_negative.sample(n=min_samples, random_state=42)
    
    # Add binary labels
    balanced_positive['binary_label'] = 1
    balanced_negative['binary_label'] = 0
    
    # Combine into final dataset
    balanced_dataset = pd.concat([balanced_positive, balanced_negative], ignore_index=True)
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    # Save balanced dataset
    balanced_output_file = 'daal_qalqalah_balanced_dataset.csv'
    balanced_dataset.to_csv(balanced_output_file, index=False)
    
    print(f"Balanced dataset created with {len(balanced_dataset)} samples:")
    print(f"  - Positive samples: {min_samples}")
    print(f"  - Negative samples: {min_samples}")
    print(f"  - Saved to: {balanced_output_file}")
    
    print("\nBalanced dataset label distribution:")
    print(balanced_dataset['binary_label'].value_counts())

print("\n" + "="*50)
print("EXTRACTION COMPLETE!")
print("\nFiles created:")
print("1. 'daal_qalqalah_positive_samples.csv' - All positive samples")
print("2. 'daal_qalqalah_negative_samples.csv' - All negative samples") 
print("3. 'daal_qalqalah_balanced_dataset.csv' - Balanced dataset for training")
print("\nNext steps for binary classification:")
print("1. Use the balanced dataset for training")
print("2. Extract audio features for both positive and negative samples")
print("3. Train your binary classifier")
print("4. Evaluate model performance")