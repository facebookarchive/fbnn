

### CachingLookupTable.lua ###

Copyright 2004-present Facebook. All Rights Reserved.

<a name="fbnn.CachingLookupTable.dok"></a>


## fbnn.CachingLookupTable ##

The lookup table itself is a hash table of Ways.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/CachingLookupTable.lua#L13">[src]</a>
<a name="fbnn.Way"></a>


### fbnn.Way(size, backingStore, statsTab) ###

A way is a fully associative portion of the cache, with fixed
capacity. Since we search it by brute-force, it needs to be
modestly sized.

<a class="entityLink" href="https://github.com/facebook/fbnn/blob/5dc9bb691436a7687026f4f39b2eea1c0b523ae8/fbnn/CachingLookupTable.lua#L259">[src]</a>
<a name="fbnn.CachingLookupTable:zeroGradParameters"></a>


### fbnn.CachingLookupTable:zeroGradParameters() ###

For now we only support the accUpdateGradParameters usage pattern.


#### Undocumented methods ####

<a name="fbnn.Way:updateRow"></a>
 * `fbnn.Way:updateRow(row, addend, lr)`
<a name="fbnn.Way:trim"></a>
 * `fbnn.Way:trim()`
<a name="fbnn.Way:flush"></a>
 * `fbnn.Way:flush()`
<a name="fbnn.Way:pull"></a>
 * `fbnn.Way:pull(row)`
<a name="fbnn.CachingLookupTable"></a>
 * `fbnn.CachingLookupTable(backingLut, numRows)`
<a name="fbnn.CachingLookupTable:flush"></a>
 * `fbnn.CachingLookupTable:flush()`
<a name="fbnn.CachingLookupTable:dumpStats"></a>
 * `fbnn.CachingLookupTable:dumpStats()`
<a name="fbnn.CachingLookupTable:notImplemented"></a>
 * `fbnn.CachingLookupTable:notImplemented(name)`
<a name="fbnn.CachingLookupTable:accUpdateOnly"></a>
 * `fbnn.CachingLookupTable:accUpdateOnly()`
<a name="fbnn.CachingLookupTable:reset"></a>
 * `fbnn.CachingLookupTable:reset(stdv)`
<a name="fbnn.CachingLookupTable:readRow"></a>
 * `fbnn.CachingLookupTable:readRow(row)`
<a name="fbnn.CachingLookupTable:writeRow"></a>
 * `fbnn.CachingLookupTable:writeRow(row, val)`
<a name="fbnn.CachingLookupTable:updateRow"></a>
 * `fbnn.CachingLookupTable:updateRow(row, val, lr)`
<a name="fbnn.CachingLookupTable:updateRows"></a>
 * `fbnn.CachingLookupTable:updateRows(rows, val)`
<a name="fbnn.CachingLookupTable:updateOutput"></a>
 * `fbnn.CachingLookupTable:updateOutput(input)`
<a name="fbnn.CachingLookupTable:accGradParameters"></a>
 * `fbnn.CachingLookupTable:accGradParameters()`
<a name="fbnn.CachingLookupTable:updateParameters"></a>
 * `fbnn.CachingLookupTable:updateParameters(lr)`
<a name="fbnn.CachingLookupTable:type"></a>
 * `fbnn.CachingLookupTable:type(type)`
<a name="fbnn.CachingLookupTable:accUpdateGradParameters"></a>
 * `fbnn.CachingLookupTable:accUpdateGradParameters(input, gradOutput, lr)`
