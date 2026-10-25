# Only changes the environment of this PowerShell session. Never writes the key to disk.
$taskSecureKey = Read-Host 'DeepSeek API Key (hidden input)' -AsSecureString
$taskKeyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecureKey)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskKeyPointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskKeyPointer)
    Remove-Variable taskSecureKey, taskKeyPointer
}
