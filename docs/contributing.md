# OrthoRoute Development Contributing Guide

## Current Status (July 2025)

### Recent Major Debugging Session
We've completed an extensive debugging session to resolve the core issue: **"OrthoRoute plugin doesn't actually route"**. The plugin would load and run without errors, but no tracks would appear on the PCB.

### Issues Identified and Fixed

#### 1. Plugin Crashes (RESOLVED ✅)
- **Problem**: Import errors and KiCad API compatibility issues
- **Solution**: Fixed import statements and API call compatibility
- **Status**: Plugin now loads without errors

#### 2. Missing Track Creation (RESOLVED ✅)
- **Problem**: Routing algorithm completed but no PCB tracks were created
- **Solution**: Implemented `_create_tracks_from_path()` method in routing engine
- **Details**: Added proper KiCad `PCB_TRACK` object creation and board integration
- **Status**: Track creation functionality now exists

#### 3. wxPython UI Compatibility (RESOLVED ✅)
- **Problem**: Dialog constructor errors in KiCad 8.0+
- **Solution**: Updated wxPython dialog initialization for current KiCad version
- **Status**: UI now displays correctly

#### 4. Net-Pad Matching Logic (RESOLVED ✅)
- **Problem**: Critical bug in net detection - plugin couldn't find nets to route
- **Root Cause**: Used object comparison (`pad.GetNet() == kicad_net`) instead of netcode comparison
- **Solution**: Changed to `pad_net.GetNetCode() == netcode` for proper matching
- **Investigation**: Created KiCad API investigation tools that revealed board had 31 nets with proper structure
- **Status**: Net matching logic corrected

### Current Challenge
Despite all fixes being applied and packaged, the plugin still reports "0 nets processed" when run. This suggests either:
1. The fixes aren't being applied correctly in the runtime environment
2. Additional net detection logic needs refinement
3. Board-specific compatibility issues remain

## How to Contribute

### Setting Up Development Environment
```bash
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute
python install_dev.py
```

### Testing Changes
```bash
# Run verification
python tests/verify_plugin.py

# Test specific functionality
python tests/test_plugin_registration.py

# Integration testing
python tests/integration_tests.py
```

### Debugging Workflow Used
1. **Systematic Issue Isolation**: Separated crashes from functional issues
2. **API Investigation**: Created tools to understand KiCad's net/pad relationships
3. **Progressive Fixes**: Applied fixes incrementally and tested each stage
4. **Package Rebuilding**: Ensured all fixes were properly packaged
5. **Runtime Verification**: Tested actual plugin behavior in KiCad

### Key Debugging Tools Created
- `board_investigator.py`: KiCad API analysis tool
- Comprehensive logging throughout routing pipeline
- Systematic error handling and reporting
- Progressive plugin versions for testing stages

### Areas Needing Contribution

#### High Priority
1. **Net Detection Refinement**: The core issue may require deeper investigation
2. **KiCad API Compatibility**: Different KiCad versions may have varying behaviors
3. **Board State Validation**: Ensuring boards are in proper state for routing

#### Medium Priority
1. **Algorithm Optimization**: Improve routing quality and speed
2. **UI Enhancements**: Better user feedback and error reporting
3. **Test Coverage**: More comprehensive automated testing

#### Documentation Priority
1. **User Troubleshooting**: Based on real-world issues encountered
2. **Developer Debugging**: Document the debugging methodology used
3. **API Reference**: KiCad integration patterns and best practices

### Debugging Best Practices Learned

#### 1. Systematic Approach
- Start with crash elimination before functional debugging
- Use comprehensive logging at each stage
- Test each fix incrementally

#### 2. API Investigation
- Create standalone tools to understand external API behavior
- Don't assume API documentation matches actual behavior
- Test with real data from target environment

#### 3. Progressive Testing
- Create minimal test cases that isolate specific functionality
- Use version control to track which fixes resolve which issues
- Verify each fix in the target environment (KiCad) not just development

#### 4. User-Centric Debugging
- Focus on the user's actual experience: "it doesn't work"
- Investigate the complete pipeline from user action to expected result
- Don't just fix errors - ensure the intended functionality actually works

### Code Style and Standards
- Follow PEP 8 for Python code
- Add comprehensive logging for debugging
- Include error handling and graceful degradation
- Document complex KiCad API usage patterns

### Pull Request Guidelines
1. **Test thoroughly** in actual KiCad environment
2. **Document debugging process** if fixing complex issues
3. **Include verification steps** for reviewers
4. **Update relevant documentation** files

### Current Development Priority
The immediate priority is resolving why the fixed plugin still reports "0 nets processed" despite all identified bugs being corrected. This likely requires:
1. Runtime debugging in KiCad environment
2. Verification that packaged fixes are properly deployed
3. Potential discovery of additional net detection edge cases

Contributors should focus on net detection logic and KiCad API integration patterns.
