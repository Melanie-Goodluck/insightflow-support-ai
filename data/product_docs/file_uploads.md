# File uploads

## Supported formats

InsightFlow accepts CSV, XLSX, and JSON files. CSV files must use UTF-8
encoding. Password-protected files and compressed ZIP archives are not
supported.

## File size and structure

Each uploaded file must be 50 MB or smaller. Every column must have a unique,
non-empty heading. A CSV file must use commas as separators and must contain at
least one data row.

## Troubleshooting rejected uploads

If an upload is rejected, check the file size, encoding, column headings, and
file format. Error E104 means that the file is larger than the 50 MB upload
limit. Error E107 means duplicate or missing column headings were detected.
Save the corrected file and upload it again.

