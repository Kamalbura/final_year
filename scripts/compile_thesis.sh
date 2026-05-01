#!/bin/bash
# Thesis Compilation Script
# Compiles all LaTeX chapters into final PDF

echo "=============================================="
echo "Thesis Compilation Script"
echo "=============================================="
echo ""

cd final_report

echo "[1/4] Checking LaTeX installation..."
if command -v pdflatex &> /dev/null; then
    echo "✓ pdflatex found"
else
    echo "✗ pdflatex not found. Please install TeX Live or MiKTeX"
    exit 1
fi

echo ""
echo "[2/4] Compiling thesis..."
# First pass
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "(Error|Warning|Output)" || true

# Run bibtex if .aux file exists
if [ -f main.aux ]; then
    echo "Running bibtex..."
    bibtex main 2>&1 | grep -E "(Error|Warning)" || true
fi

# Second pass for references
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "(Error|Warning)" || true

# Final pass
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "(Error|Warning|Output)" || true

echo ""
echo "[3/4] Checking output..."
if [ -f main.pdf ]; then
    echo "✓ PDF generated successfully"
    ls -lh main.pdf
else
    echo "✗ PDF generation failed"
    exit 1
fi

echo ""
echo "[4/4] Copying to outputs..."
cp main.pdf ../outputs/thesis_final.pdf
echo "✓ Copied to outputs/thesis_final.pdf"

echo ""
echo "=============================================="
echo "Compilation Complete!"
echo "=============================================="
echo ""
echo "Final thesis: final_report/main.pdf"
echo "Copy: outputs/thesis_final.pdf"
