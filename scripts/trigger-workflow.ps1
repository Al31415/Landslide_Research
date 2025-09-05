$headers = @{
    "Accept" = "application/vnd.github+json"
    "Authorization" = "Bearer $env:GITHUB_PAT"
    "X-GitHub-Api-Version" = "2022-11-28"
}

if ([string]::IsNullOrWhiteSpace($env:GITHUB_PAT)) {
    Write-Error "GITHUB_PAT environment variable is not set. Set it with your GitHub token before running."
    exit 1
}

$body = @{
    ref = "deploy-app"
} | ConvertTo-Json

Write-Output "Triggering deployment workflow..."

$response = Invoke-RestMethod -Uri "https://api.github.com/repos/Al31415/Landslide_Research/actions/workflows/deploy-fly.yml/dispatches" `
                  -Method Post `
                  -Headers $headers `
                  -Body $body `
                  -ContentType "application/json"

if ($response -eq $null) {
    Write-Output "✅ Workflow triggered successfully! Check GitHub Actions for progress."
} else {
    Write-Output "Response: $response"
} 