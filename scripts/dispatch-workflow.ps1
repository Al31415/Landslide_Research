param(
    [Parameter(Mandatory=$true)] [string]$Workflow,
    [Parameter(Mandatory=$true)] [string]$Ref
)

if ([string]::IsNullOrWhiteSpace($env:GITHUB_PAT)) {
    Write-Error "GITHUB_PAT environment variable is not set. Set it with your GitHub token before running."
    exit 1
}

$headers = @{
    "Accept" = "application/vnd.github+json"
    "Authorization" = "Bearer $env:GITHUB_PAT"
    "X-GitHub-Api-Version" = "2022-11-28"
}

$body = @{ ref = $Ref } | ConvertTo-Json

$owner = "Al31415"
$repo = "Landslide_Research"
$uri = "https://api.github.com/repos/$owner/$repo/actions/workflows/$Workflow/dispatches"

Write-Output "Dispatching workflow '$Workflow' on ref '$Ref'..."

$response = Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -Body $body -ContentType "application/json" -ErrorAction Stop

if ($response -eq $null) {
    Write-Output "✅ Workflow '$Workflow' dispatched successfully. Check GitHub Actions for progress."
} else {
    Write-Output "Response: $response"
} 