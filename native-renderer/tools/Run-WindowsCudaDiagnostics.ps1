param(
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$packageRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$nativeRendererRoot = Join-Path $packageRoot 'native-renderer'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$scratchRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("imprint-cuda-diagnostics-{0}-{1}" -f $timestamp, ([guid]::NewGuid().ToString('N').Substring(0, 8)))
$reportRoot = Join-Path $scratchRoot 'report'
$logsRoot = Join-Path $reportRoot 'logs'
$buildRoot = Join-Path $scratchRoot 'build'

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $archivePath = Join-Path $packageRoot ("cuda-diagnostics-{0}.zip" -f $timestamp)
} elseif ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $archivePath = [System.IO.Path]::GetFullPath($OutputPath)
} else {
    $archivePath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $OutputPath))
}

if (Test-Path -LiteralPath $archivePath) {
    throw "Refusing to overwrite an existing diagnostics archive: $archivePath"
}

New-Item -ItemType Directory -Path $logsRoot -Force | Out-Null

function Get-ApplicationPath {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $command) { return $null }
    if (-not [string]::IsNullOrWhiteSpace($command.Source)) { return $command.Source }
    return $command.Definition
}

function Invoke-LoggedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$LogPath
    )
    "COMMAND: $Executable $($Arguments -join ' ')" | Out-File -LiteralPath $LogPath -Encoding utf8
    & $Executable @Arguments 2>&1 | Out-File -LiteralPath $LogPath -Append -Encoding utf8
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) { $exitCode = 0 }
    "EXIT_CODE: $exitCode" | Out-File -LiteralPath $LogPath -Append -Encoding utf8
    return [int]$exitCode
}

function Write-TextFile {
    param([string]$Path, [string[]]$Lines)
    $Lines | Out-File -LiteralPath $Path -Encoding utf8
}

function Format-ExitStatus {
    param([string]$Name, [object]$ExitCode, [string]$WhenMissing = 'NOT RUN')
    if ($null -eq $ExitCode) { return "$Name`: $WhenMissing" }
    $code = [int]$ExitCode
    if ($code -eq 0) { return "$Name`: PASS (exit code 0)" }
    if ($code -eq 77) { return "$Name`: SKIP (exit code 77)" }
    return "$Name`: FAIL (exit code $code)"
}

function Sanitize-ReportFiles {
    $replacements = @(
        @{ Value = $packageRoot; Replacement = '[PACKAGE_ROOT]' },
        @{ Value = $scratchRoot; Replacement = '[TEMP_BUILD_ROOT]' },
        @{ Value = $env:USERPROFILE; Replacement = '[USERPROFILE]' },
        @{ Value = $env:TEMP; Replacement = '[TEMP]' },
        @{ Value = $env:TMP; Replacement = '[TEMP]' }
    )
    $files = Get-ChildItem -LiteralPath $reportRoot -File -Recurse
    foreach ($file in $files) {
        $content = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($null -eq $content) { continue }
        foreach ($item in $replacements) {
            if (-not [string]::IsNullOrWhiteSpace($item.Value)) {
                $content = $content.Replace($item.Value, $item.Replacement)
            }
        }
        Set-Content -LiteralPath $file.FullName -Value $content -Encoding utf8 -NoNewline
    }
}

$cmakeExit = $null
$buildExit = $null
$ctestExit = $null
$benchmarkExit = $null
$pythonCheckExit = $null
$pythonTestExit = $null
$basicBridgeCheckExit = $null
$basicBridgeTestExit = $null
$gpuContractStatus = 'NOT RUN'
$abiStatus = 'NOT RUN'
$pythonStatus = 'NOT RUN'
$basicBridgeStatus = 'NOT RUN'
$cmakePath = Get-ApplicationPath 'cmake.exe'
$ctestPath = Get-ApplicationPath 'ctest.exe'
$nvccPath = Get-ApplicationPath 'nvcc.exe'
$nvidiaSmiPath = Get-ApplicationPath 'nvidia-smi.exe'
$pythonPath = Get-ApplicationPath 'python.exe'
$msvcPath = Get-ApplicationPath 'cl.exe'

try {
    $environmentLines = [System.Collections.Generic.List[string]]::new()
    $environmentLines.Add('Windows x64 CUDA native renderer diagnostics')
    $environmentLines.Add(('Collected UTC: {0}' -f [DateTime]::UtcNow.ToString('yyyy-MM-dd HH:mm:ss')))
    $environmentLines.Add(('PowerShell version: {0}' -f $PSVersionTable.PSVersion.ToString()))
    $environmentLines.Add(('64-bit OS process: {0}' -f [Environment]::Is64BitOperatingSystem))
    $environmentLines.Add(('64-bit PowerShell process: {0}' -f [Environment]::Is64BitProcess))
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        $environmentLines.Add(('Windows: {0}; version {1}; build {2}; architecture {3}' -f $os.Caption, $os.Version, $os.BuildNumber, $os.OSArchitecture))
    } catch {
        $environmentLines.Add(('Windows: unavailable ({0})' -f $_.Exception.Message))
    }
    try {
        $processor = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop | Select-Object -First 1
        if ($null -ne $processor) {
            $environmentLines.Add(('CPU: {0}; cores {1}; architecture code {2}' -f $processor.Name, $processor.NumberOfCores, $processor.Architecture))
        }
    } catch {
        $environmentLines.Add(('CPU: unavailable ({0})' -f $_.Exception.Message))
    }
    try {
        $videoControllers = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop)
        if ($videoControllers.Count -eq 0) {
            $environmentLines.Add('GPU (Windows video controller): none reported')
        } else {
            foreach ($gpu in $videoControllers) {
                $environmentLines.Add(('GPU: {0}; vendor {1}; driver {2}; driver date {3}' -f $gpu.Name, $gpu.AdapterCompatibility, $gpu.DriverVersion, $gpu.DriverDate))
            }
        }
    } catch {
        $environmentLines.Add(('GPU (Windows video controller): unavailable ({0})' -f $_.Exception.Message))
    }
    Write-TextFile (Join-Path $logsRoot 'environment.txt') $environmentLines.ToArray()

    if ($null -ne $nvidiaSmiPath) {
        $null = Invoke-LoggedCommand $nvidiaSmiPath @('--query-gpu=name,driver_version', '--format=csv,noheader') (Join-Path $logsRoot 'nvidia-smi.txt')
    } else {
        Write-TextFile (Join-Path $logsRoot 'nvidia-smi.txt') @('nvidia-smi.exe was not found on PATH.')
    }

    if ($null -ne $nvccPath) {
        $null = Invoke-LoggedCommand $nvccPath @('--version') (Join-Path $logsRoot 'nvcc.txt')
    } else {
        Write-TextFile (Join-Path $logsRoot 'nvcc.txt') @('nvcc.exe was not found on PATH.')
    }

    if ($null -ne $cmakePath) {
        $null = Invoke-LoggedCommand $cmakePath @('--version') (Join-Path $logsRoot 'cmake-version.txt')
    } else {
        Write-TextFile (Join-Path $logsRoot 'cmake-version.txt') @('cmake.exe was not found on PATH.')
    }

    $msvcLines = [System.Collections.Generic.List[string]]::new()
    $msvcLines.Add(('cl.exe on PATH: {0}' -f ($null -ne $msvcPath)))
    if ($null -ne $msvcPath) {
        $null = Invoke-LoggedCommand $msvcPath @('/Bv') (Join-Path $logsRoot 'msvc-cl-version.txt')
    } else {
        Write-TextFile (Join-Path $logsRoot 'msvc-cl-version.txt') @('cl.exe was not found on PATH; this is common outside a Visual Studio Developer PowerShell.')
    }
    $vswherePath = $null
    if (-not [string]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
        $candidate = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (Test-Path -LiteralPath $candidate) { $vswherePath = $candidate }
    }
    if ($null -ne $vswherePath) {
        $vsVersionLog = Join-Path $logsRoot 'visual-studio-version.txt'
        $vswhereExit = Invoke-LoggedCommand $vswherePath @('-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationVersion') $vsVersionLog
        if ($vswhereExit -eq 0) {
            $msvcLines.Add('Visual Studio C++ toolset discovered by vswhere: yes (version in visual-studio-version.txt)')
        } else {
            $msvcLines.Add('Visual Studio C++ toolset discovered by vswhere: no')
        }
    } else {
        $msvcLines.Add('Visual Studio C++ toolset discovery: vswhere.exe not found at its standard installer location')
    }
    Write-TextFile (Join-Path $logsRoot 'msvc.txt') $msvcLines.ToArray()

    if (-not (Test-Path -LiteralPath (Join-Path $nativeRendererRoot 'CMakeLists.txt')) -or
        -not (Test-Path -LiteralPath (Join-Path $packageRoot 'src\dehaze.py'))) {
        Write-TextFile (Join-Path $logsRoot 'package-layout.txt') @(
            'Expected extracted package layout was not found.',
            'Expected native-renderer/CMakeLists.txt and src/dehaze.py below the package root.'
        )
    } else {
        Write-TextFile (Join-Path $logsRoot 'package-layout.txt') @('Expected package layout found: native-renderer/ and src/dehaze.py.')
    }

    if ($null -ne $cmakePath -and
        (Test-Path -LiteralPath (Join-Path $nativeRendererRoot 'CMakeLists.txt')) -and
        (Test-Path -LiteralPath (Join-Path $packageRoot 'src\dehaze.py'))) {
        $cmakeExit = Invoke-LoggedCommand $cmakePath @(
            '-S', $nativeRendererRoot,
            '-B', $buildRoot,
            '-DIMPRINT_ENABLE_CUDA=ON',
            '-DBUILD_TESTING=ON',
            '-DCMAKE_BUILD_TYPE=Release'
        ) (Join-Path $logsRoot 'cmake-configure.log')

        $cachePath = Join-Path $buildRoot 'CMakeCache.txt'
        $configLines = [System.Collections.Generic.List[string]]::new()
        $configLines.Add('Selected CMake configuration values (full CMakeCache.txt is excluded).')
        if (Test-Path -LiteralPath $cachePath) {
            $cacheLines = Get-Content -LiteralPath $cachePath
            $keys = @(
                'CMAKE_GENERATOR',
                'CMAKE_BUILD_TYPE',
                'BUILD_TESTING',
                'IMPRINT_ENABLE_CUDA',
                'CMAKE_CUDA_COMPILER_ID',
                'CMAKE_CUDA_COMPILER_VERSION',
                'CMAKE_CUDA_ARCHITECTURES',
                'CMAKE_CXX_COMPILER_ID',
                'CMAKE_CXX_COMPILER_VERSION'
            )
            foreach ($key in $keys) {
                $match = $cacheLines | Where-Object { $_ -match ('^' + [regex]::Escape($key) + ':[^=]*=') } | Select-Object -First 1
                if ($null -ne $match) {
                    $value = ($match -split '=', 2)[1]
                    $configLines.Add(('{0}={1}' -f $key, $value))
                } else {
                    $configLines.Add(('{0}=<not set>' -f $key))
                }
            }
            $cudaCompiler = $cacheLines | Where-Object { $_ -match '^CMAKE_CUDA_COMPILER:[^=]*=' } | Select-Object -First 1
            if ($null -ne $cudaCompiler) {
                $compilerValue = ($cudaCompiler -split '=', 2)[1]
                if (-not [string]::IsNullOrWhiteSpace($compilerValue)) {
                    $configLines.Add(('CMAKE_CUDA_COMPILER_BASENAME={0}' -f [System.IO.Path]::GetFileName($compilerValue)))
                }
            }
        } else {
            $configLines.Add('CMake did not produce a cache file.')
        }
        Write-TextFile (Join-Path $reportRoot 'build-config-summary.txt') $configLines.ToArray()

        if ($cmakeExit -eq 0) {
            $buildExit = Invoke-LoggedCommand $cmakePath @('--build', $buildRoot, '--config', 'Release', '--parallel') (Join-Path $logsRoot 'cmake-build.log')
        }

        if ($buildExit -eq 0 -and $null -ne $ctestPath) {
            $junitPath = Join-Path $logsRoot 'ctest-results.xml'
            $ctestExit = Invoke-LoggedCommand $ctestPath @(
                '--test-dir', $buildRoot,
                '-C', 'Release',
                '--output-on-failure',
                '--output-junit', $junitPath
            ) (Join-Path $logsRoot 'ctest.log')

            if (Test-Path -LiteralPath $junitPath) {
                try {
                    [xml]$ctestXml = Get-Content -LiteralPath $junitPath -Raw
                    $gpuCase = $ctestXml.SelectSingleNode("//testcase[@name='native_renderer_gpu_contract']")
                    $abiCase = $ctestXml.SelectSingleNode("//testcase[@name='native_renderer_abi_validation']")
                    if ($null -ne $gpuCase) {
                        if ($null -ne $gpuCase.SelectSingleNode('./skipped')) {
                            $gpuContractStatus = 'SKIP (CTest classified SKIP_RETURN_CODE=77; CUDA contract did not pass)'
                        } elseif ($null -ne $gpuCase.SelectSingleNode('./failure') -or $null -ne $gpuCase.SelectSingleNode('./error')) {
                            $gpuContractStatus = 'FAIL (CTest reported a GPU contract failure)'
                        } else {
                            $gpuContractStatus = 'PASS'
                        }
                    }
                    if ($null -ne $abiCase) {
                        if ($null -ne $abiCase.SelectSingleNode('./failure') -or $null -ne $abiCase.SelectSingleNode('./error')) {
                            $abiStatus = 'FAIL'
                        } elseif ($null -ne $abiCase.SelectSingleNode('./skipped')) {
                            $abiStatus = 'SKIP'
                        } else {
                            $abiStatus = 'PASS'
                        }
                    }
                } catch {
                    $gpuContractStatus = 'UNKNOWN (could not parse CTest JUnit output)'
                    $abiStatus = 'UNKNOWN (could not parse CTest JUnit output)'
                }
            }
        } elseif ($buildExit -eq 0) {
            Write-TextFile (Join-Path $logsRoot 'ctest.log') @('ctest.exe was not found on PATH; tests were not run.')
        }

        if ($buildExit -eq 0) {
            $benchmarkPath = Join-Path $buildRoot 'Release\imprint_renderer_benchmark.exe'
            if (-not (Test-Path -LiteralPath $benchmarkPath)) {
                $benchmarkPath = Join-Path $buildRoot 'imprint_renderer_benchmark.exe'
            }
            if (Test-Path -LiteralPath $benchmarkPath) {
                $benchmarkExit = Invoke-LoggedCommand $benchmarkPath @() (Join-Path $logsRoot 'l2-benchmark.log')
            } else {
                Write-TextFile (Join-Path $logsRoot 'l2-benchmark.log') @('CMake build did not produce imprint_renderer_benchmark.exe.')
            }
        }

        if ($buildExit -eq 0) {
            $rendererDll = Get-ChildItem -LiteralPath $buildRoot -Filter 'imprint_renderer.dll' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -eq $pythonPath) {
                $pythonStatus = 'SKIP (python.exe was not found on PATH)'
                Write-TextFile (Join-Path $logsRoot 'python-availability.log') @('python.exe was not found on PATH; NumPy/pytest availability could not be checked.')
            } else {
                $pythonCheckExit = Invoke-LoggedCommand $pythonPath @('-c', 'import sys, numpy, pytest; print("python=" + sys.version.split()[0]); print("numpy=" + numpy.__version__); print("pytest=" + pytest.__version__)') (Join-Path $logsRoot 'python-availability.log')
                if ($pythonCheckExit -ne 0) {
                    $pythonStatus = 'SKIP (current PATH Python does not have both numpy and pytest)'
                } elseif ($null -eq $rendererDll) {
                    $pythonStatus = 'SKIP (CMake-built imprint_renderer.dll was not found)'
                } else {
                    $pythonTestPath = Join-Path $nativeRendererRoot 'tests\test_metal_dehaze.py'
                    $oldLibrary = $env:IMPRINT_NATIVE_RENDERER_LIB
                    $oldBackend = $env:IMPRINT_NATIVE_RENDERER_BACKEND
                    $oldPath = $env:PATH
                    try {
                        $env:IMPRINT_NATIVE_RENDERER_LIB = $rendererDll.FullName
                        $env:IMPRINT_NATIVE_RENDERER_BACKEND = 'CUDA'
                        $env:PATH = $rendererDll.DirectoryName + [System.IO.Path]::PathSeparator + $oldPath
                        Push-Location $packageRoot
                        try {
                            $pythonTestExit = Invoke-LoggedCommand $pythonPath @('-m', 'pytest', '-q', '-s', $pythonTestPath) (Join-Path $logsRoot 'python-cuda-comparison.log')
                        } finally {
                            Pop-Location
                        }
                    } finally {
                        $env:IMPRINT_NATIVE_RENDERER_LIB = $oldLibrary
                        $env:IMPRINT_NATIVE_RENDERER_BACKEND = $oldBackend
                        $env:PATH = $oldPath
                    }
                    if ($pythonTestExit -eq 0) {
                        $pythonStatus = 'PASS (CUDA shader compared with Python CPU reference)'
                    } else {
                        $pythonStatus = (Format-ExitStatus 'Python CUDA comparison' $pythonTestExit)
                    }
                }

                $basicBridgeTestPath = Join-Path $nativeRendererRoot 'tests\test_basic_bridge.py'
                $basicBridgeRequiredFiles = @(
                    (Join-Path $packageRoot 'src\native_renderer.py'),
                    (Join-Path $packageRoot 'src\ricoh_filter.py'),
                    (Join-Path $packageRoot 'src\image_io.py'),
                    $basicBridgeTestPath
                )
                $basicBridgeFilesPresent = $true
                foreach ($requiredFile in $basicBridgeRequiredFiles) {
                    if (-not (Test-Path -LiteralPath $requiredFile)) { $basicBridgeFilesPresent = $false }
                }
                if (-not $basicBridgeFilesPresent) {
                    $basicBridgeStatus = 'SKIP (basic bridge test or one of its required source files is absent from this package)'
                    Write-TextFile (Join-Path $logsRoot 'python-basic-bridge-availability.log') @('The basic bridge suite requires src/native_renderer.py, src/ricoh_filter.py, src/image_io.py, and native-renderer/tests/test_basic_bridge.py; at least one was not present.')
                } elseif ($pythonCheckExit -ne 0) {
                    $basicBridgeStatus = 'SKIP (current PATH Python does not have both numpy and pytest)'
                    Write-TextFile (Join-Path $logsRoot 'python-basic-bridge-availability.log') @('Basic bridge comparison was not attempted because NumPy/pytest were unavailable; see python-availability.log.')
                } else {
                    $basicBridgeCheckExit = Invoke-LoggedCommand $pythonPath @('-c', 'import cv2, PIL, rawpy; print("opencv=" + cv2.__version__); print("pillow=" + PIL.__version__); print("rawpy=" + rawpy.__version__)') (Join-Path $logsRoot 'python-basic-bridge-availability.log')
                    if ($basicBridgeCheckExit -ne 0) {
                        $basicBridgeStatus = 'SKIP (current PATH Python does not have cv2, Pillow, and rawpy)'
                    } elseif ($null -eq $rendererDll) {
                        $basicBridgeStatus = 'SKIP (CMake-built imprint_renderer.dll was not found)'
                    } else {
                        $oldLibrary = $env:IMPRINT_NATIVE_RENDERER_LIB
                        $oldBackend = $env:IMPRINT_NATIVE_RENDERER_BACKEND
                        $oldPath = $env:PATH
                        try {
                            $env:IMPRINT_NATIVE_RENDERER_LIB = $rendererDll.FullName
                            $env:IMPRINT_NATIVE_RENDERER_BACKEND = 'CUDA'
                            $env:PATH = $rendererDll.DirectoryName + [System.IO.Path]::PathSeparator + $oldPath
                            Push-Location $packageRoot
                            try {
                                $basicBridgeTestExit = Invoke-LoggedCommand $pythonPath @('-m', 'pytest', '-q', '-s', $basicBridgeTestPath) (Join-Path $logsRoot 'python-basic-bridge.log')
                            } finally {
                                Pop-Location
                            }
                        } finally {
                            $env:IMPRINT_NATIVE_RENDERER_LIB = $oldLibrary
                            $env:IMPRINT_NATIVE_RENDERER_BACKEND = $oldBackend
                            $env:PATH = $oldPath
                        }
                        if ($basicBridgeTestExit -eq 0) {
                            $basicBridgeLog = Get-Content -LiteralPath (Join-Path $logsRoot 'python-basic-bridge.log') -Raw -ErrorAction SilentlyContinue
                            if ($basicBridgeLog -match '(?im)\b\d+\s+skipped\b') {
                                $basicBridgeStatus = 'SKIP/PARTIAL (pytest reported skipped cases; see python-basic-bridge.log)'
                            } else {
                                $basicBridgeStatus = 'PASS (native basic adjustments compared with Python preview reference)'
                            }
                        } else {
                            $basicBridgeStatus = (Format-ExitStatus 'Python basic bridge comparison' $basicBridgeTestExit)
                        }
                    }
                }
            }
        } else {
            $pythonStatus = 'NOT RUN (native renderer build did not succeed)'
            $basicBridgeStatus = 'NOT RUN (native renderer build did not succeed)'
            Write-TextFile (Join-Path $logsRoot 'python-availability.log') @('Python comparison was not attempted because the native renderer build did not succeed.')
            Write-TextFile (Join-Path $logsRoot 'python-basic-bridge-availability.log') @('Basic bridge comparison was not attempted because the native renderer build did not succeed.')
        }
    } else {
        Write-TextFile (Join-Path $reportRoot 'build-config-summary.txt') @('CMake configure was not attempted because cmake.exe or the expected package layout was unavailable.')
        if ($null -eq $cmakePath) {
            Write-TextFile (Join-Path $logsRoot 'cmake-configure.log') @('cmake.exe was not found on PATH; configure/build/tests/benchmark were not run.')
        }
        $pythonStatus = 'NOT RUN (CMake configure was not attempted)'
    }
} catch {
    Write-TextFile (Join-Path $logsRoot 'script-error.log') @($_.Exception.Message)
} finally {
    $summaryLines = [System.Collections.Generic.List[string]]::new()
    $summaryLines.Add('Windows x64 CUDA native renderer diagnostics')
    $summaryLines.Add(('Collected UTC: {0}' -f [DateTime]::UtcNow.ToString('yyyy-MM-dd HH:mm:ss')))
    if ($null -ne $cmakeExit) {
        $summaryLines.Add((Format-ExitStatus 'CMake configure' $cmakeExit))
    } elseif ($null -eq $cmakePath) {
        $summaryLines.Add('CMake configure: NOT RUN (cmake.exe unavailable)')
    } else {
        $summaryLines.Add('CMake configure: NOT RUN (expected package layout unavailable or script stopped early)')
    }
    $summaryLines.Add((Format-ExitStatus 'CMake build' $buildExit))
    if ($null -ne $ctestExit) {
        $summaryLines.Add((Format-ExitStatus 'CTest suite' $ctestExit))
    } else {
        $summaryLines.Add('CTest suite: NOT RUN')
    }
    $summaryLines.Add(('CTest GPU contract: {0}' -f $gpuContractStatus))
    $summaryLines.Add(('CTest ABI validation: {0}' -f $abiStatus))
    if ($benchmarkExit -eq 77) {
        $summaryLines.Add('L2 benchmark: SKIP (exit code 77; no usable GPU backend)')
    } elseif ($null -ne $benchmarkExit) {
        $summaryLines.Add((Format-ExitStatus 'L2 benchmark' $benchmarkExit))
    } else {
        $summaryLines.Add('L2 benchmark: NOT RUN')
    }
    $summaryLines.Add(('Python CUDA comparison: {0}' -f $pythonStatus))
    $summaryLines.Add(('Python basic bridge comparison: {0}' -f $basicBridgeStatus))
    $summaryLines.Add('')
    $summaryLines.Add('Archive contents are limited to this summary, logs, and selected build configuration values.')
    $summaryLines.Add('The archive excludes the CMake build directory, binaries, full CMakeCache, source photos, and user files.')
    Write-TextFile (Join-Path $reportRoot 'summary.txt') $summaryLines.ToArray()

    Sanitize-ReportFiles
    $archiveDirectory = Split-Path -Parent $archivePath
    if (-not (Test-Path -LiteralPath $archiveDirectory)) {
        New-Item -ItemType Directory -Path $archiveDirectory -Force | Out-Null
    }
    Compress-Archive -Path (Join-Path $reportRoot '*') -DestinationPath $archivePath -CompressionLevel Optimal
}

Remove-Item -LiteralPath $scratchRoot -Recurse -Force
Write-Output ("Created diagnostics archive: {0}" -f $archivePath)

$diagnosticExitCode = 0
if ($null -eq $cmakeExit -or $cmakeExit -ne 0 -or
    $null -eq $buildExit -or $buildExit -ne 0 -or
    $null -eq $ctestExit -or $ctestExit -ne 0 -or
    $null -eq $benchmarkExit -or
    ($null -ne $benchmarkExit -and $benchmarkExit -ne 0 -and $benchmarkExit -ne 77) -or
    $gpuContractStatus -eq 'NOT RUN' -or
    $gpuContractStatus.StartsWith('FAIL') -or
    $gpuContractStatus.StartsWith('UNKNOWN') -or
    ($null -ne $pythonTestExit -and $pythonTestExit -ne 0) -or
    ($null -ne $basicBridgeTestExit -and $basicBridgeTestExit -ne 0)) {
    $diagnosticExitCode = 1
} elseif ($gpuContractStatus.StartsWith('SKIP') -or $benchmarkExit -eq 77) {
    $diagnosticExitCode = 2
}

exit $diagnosticExitCode
