param(
    [Parameter(Mandatory=$true)][string]$GitHubUser,
    [Parameter(Mandatory=$true)][string]$Repo,
    [Parameter(Mandatory=$true)][string]$Pat,
    [string]$WorkflowFile = 'build-push-docker.yml',
    [string]$Ref = 'deploy-app'
)

$ErrorActionPreference = 'Stop'
$headers = @{ Authorization = "Bearer $Pat"; Accept = 'application/vnd.github+json' }
$uri = "https://api.github.com/repos/$GitHubUser/$Repo/actions/workflows/$WorkflowFile/dispatches"
$body = @{ ref = $Ref } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri $uri -Headers $headers -Body $body | Out-Null
Write-Host "Workflow dispatch sent for $GitHubUser/$Repo@$Ref ($WorkflowFile)" 