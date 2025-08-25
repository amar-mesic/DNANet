import pandas as pd
import os

def individualize_genotypes(
    input_path: str,
    file_type: str,
    exclude_columns: list,
    output_dir: str
):
    """
    Convert genotype data from a file (Excel or CSV) into individual sample files.
    Each sample will be saved as a separate CSV file with alleles split into two columns.
    :param input_path: Path to the input file (Excel or CSV).
    :param file_type: Type of the input file, either 'excel' or 'csv'.
    :param exclude_columns: List of columns to exclude from the genotype data. 
        NOTE: Set the first column as the sample ID (e.g., 'Sample ID')!
    :param output_dir: Directory where the output files will be saved.
    :raises ValueError: If file_type is not 'excel' or 'csv'.
    """

    # Read the input file
    if file_type.lower() == "excel":
        df = pd.read_excel(input_path)
    elif file_type.lower() == "csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("file_type must be 'excel' or 'csv'")

    # Check output directory exists
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist. Please create it yourself first to ensure you place it in the right place.")

    # Determine marker columns
    marker_columns = [col for col in df.columns if col not in exclude_columns]

    # For each sample (row) in the DataFrame:
    for idx, row in df.iterrows():
        sample_name = row[exclude_columns[0]]  # e.g., 'Sample ID'
        data = []
        for marker in marker_columns:
            #skip the marker if the value is NaN or empty
            if pd.isna(row[marker]) or row[marker] == "":
                continue
            alleles = str(row[marker]).split(',')
            allele1 = alleles[0] if len(alleles) > 0 else ''
            allele2 = alleles[1] if len(alleles) > 1 else ''
            data.append([sample_name, marker, allele1, allele2])
        new_df = pd.DataFrame(data, columns=[exclude_columns[0], 'Marker', 'Allele1', 'Allele2'])
        out_path = os.path.join(output_dir, f"{sample_name}.csv")
        new_df.to_csv(out_path, sep=';', index=False)