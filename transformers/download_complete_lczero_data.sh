#!/bin/bash

# Flag to track if an interrupt occurred
interrupt_received=false

# Handle keyboard interrupt (Ctrl+C)
interrupt_handler() {
    echo -e "\n\nInterrupt received! Skipping remaining downloads and proceeding to extraction phase."
    interrupt_received=true
}

# Set up the trap to catch SIGINT (Ctrl+C)
trap interrupt_handler SIGINT

# Create directories if they don't exist
mkdir -p data-dir/raw

# URL base
BASE_URL="https://storage.lczero.org/files/training_data/run3"

# Start and end dates in YYYYMMDD-HHMM format
START_DATE="20190614-2118"
END_DATE="20190614-2318"  # Changed to just download 2 hours worth of data

# Function to get next timestamp
get_next_timestamp() {
    local current="$1"
    local year=${current:0:4}
    local month=${current:4:2}
    local day=${current:6:2}
    local hour=${current:9:2}
    local minute=${current:11:2}
    
    # Add 1 hour
    hour=$((10#$hour + 1))
    
    # Handle hour overflow
    if [ $hour -eq 24 ]; then
        hour=0
        # Add 1 day
        day=$((10#$day + 1))
        
        # Handle day overflow (simplified - assuming all months have 31 days)
        if [ $day -gt 31 ]; then
            day=1
            month=$((10#$month + 1))
            
            # Handle month overflow
            if [ $month -gt 12 ]; then
                month=1
                year=$((10#$year + 1))
            fi
        fi
    fi
    
    # Format back to YYYYMMDD-HHMM
    printf "%04d%02d%02d-%02d%02d" $year $month $day $hour $minute
}

# Function to check if current date is past end date
is_past_end_date() {
    local current="$1"
    local end="$2"
    
    # Simple string comparison to check if we've gone past the end date
    if [[ "$current" > "$end" ]]; then
        return 0  # true in bash
    else
        return 1  # false in bash
    fi
}

echo "Starting download of Leela Chess Zero training data files..."
echo "This will download files from $START_DATE to $END_DATE"
echo "Press Ctrl+C at any time to skip remaining downloads and proceed to extraction"
echo "All data will be extracted to data-dir/raw"

# Start with the first date
current_date=$START_DATE
download_count=0
skipped_count=0
failed_downloads=0

# Download files sequentially
while true; do
    # Check if interrupt was received
    if $interrupt_received; then
        echo "Skipping remaining downloads due to user interrupt"
        break
    fi
    
    filename="training-run3-${current_date}.tar"
    file_url="${BASE_URL}/${filename}"
    output_path="data-dir/${filename}"
    
    # Check if the file already exists
    if [ -f "$output_path" ]; then
        echo "Skipping $filename (already downloaded)"
        skipped_count=$((skipped_count + 1))
    else
        echo "Downloading $filename..."
        
        # Try to download the file
        if wget -q --show-progress -P data-dir "${file_url}"; then
            echo "  Download successful"
            download_count=$((download_count + 1))
            failed_downloads=0  # Reset failed counter after successful download
        else
            echo "  WARNING: Failed to download $filename, moving to next file"
            failed_downloads=$((failed_downloads + 1))
            
            # If we have more than 5 consecutive failures, we might be at the end
            if [ $failed_downloads -gt 5 ]; then
                echo "Too many failed downloads in a row, stopping download phase"
                break
            fi
        fi
    fi
    
    # Check if we've reached the end date
    if [ "$current_date" = "$END_DATE" ]; then
        echo "Reached target date $END_DATE, stopping download phase"
        break
    fi
    
    # Get next date and time
    current_date=$(get_next_timestamp "$current_date")
    
    # Safety check - if we've gone past the end date, stop
    if is_past_end_date "$current_date" "$END_DATE"; then
        echo "Generated date $current_date is past end date $END_DATE, stopping download phase"
        break
    fi
done

# Reset the trap to default behavior for the rest of the script
trap - SIGINT

echo "Download phase complete."
echo "Downloaded: $download_count new tar files"
echo "Skipped: $skipped_count already existing tar files"

# Extract tar files to raw directory
echo "Extracting tar files to data-dir/raw..."
for tar_file in data-dir/training-run3-*.tar; do
    if [ -f "$tar_file" ]; then
        # Check if extracted files already exist
        tar_basename=$(basename "$tar_file" .tar)
        # Check if any files from this tar already exist in raw directory
        if ls data-dir/raw/$tar_basename*.gz >/dev/null 2>&1; then
            echo "Skipping extraction of $tar_file (files already extracted)"
        else
            echo "Extracting $tar_file..."
            # Use --no-same-owner to avoid permission errors
            tar --no-same-owner -xf "$tar_file" -C data-dir/raw
        fi
    fi
done

# Count the files
total_files=$(find data-dir/raw -name "*.gz" | wc -l)

echo "Data processing complete!"
echo "Total .gz files in data-dir/raw: $total_files"
echo "Files are ready for training with Chessformer!"
echo ""
echo "To use this data, run the training script:"
echo "python run.py train --config configs/small_model_config.yaml --data_dir data-dir --output_dir models/small/run1" 