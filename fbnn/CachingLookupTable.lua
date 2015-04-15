-- Copyright 2004-present Facebook. All Rights Reserved.

local pl = require('pl.import_into')()
local util = require('fb.util')

-- Writeback cache of a (perhaps distributed) LookupTable.
-- CachingLookupTable implements the same interface as a LookupTable,
-- so they can compose.

local dprint = require('fb.util.dbg').new('CLuT')

-- A way is a fully associative portion of the cache, with fixed
-- capacity. Since we search it by brute-force, it needs to be
-- modestly sized.
local Way = pl.class()
function Way:_init(size, backingStore, statsTab)
    assert(size)
    assert(backingStore)
    assert(statsTab)
    dprint("creating Way of size ", size)
    self.backing = backingStore
    self.size = size
    self.stats = statsTab
    -- self.rows maps indices in the underlying lookup table to
    -- possibly-stale local copies of their value.
    self.rows = { }

    -- We have no O(1) way to count the number of entries in a raw lua
    -- table, so track size explicitly.
    self.numRows = 0

    -- self.bufferedGrads' keyspace is a subset of self.cache's.
    -- It contains the accumulated local updates for each modified
    -- row. A dirty row is one that is present in self.cache,
    -- and has an entry in self.bufferedGrads.
    self.bufferedGrads = { }
end

function Way:_incStat(name)
    self.stats[name] = self.stats[name] + 1
end

function Way:updateRow(row, addend, lr)
    local lr = lr or 1.0
    local found = false
    dprint("looking for row", row)
    if self.rows[row] then
        found = true
        dprint("gradUp!", self.rows[row], addend)
        self.rows[row]:add(-lr, addend)
        if not self.bufferedGrads[row] then
            self.bufferedGrads[row] = torch.zeros(addend:nElement())
        end
        self.bufferedGrads[row]:add(-lr, addend)
        return
    end
    if not found then
        -- Possible if we evicted in the normal course of doing business,
        -- or if we wrote rather than read. We could write this through,
        -- but let's just read it in.
        dprint("missed looking for row", row)
        self:pull(row)
        assert(self.rows[row])
        self:updateRow(row, addend, lr)
    end
end

function Way:_writeBackOne(row)
    if self.bufferedGrads[row] then
        assert(self.rows[row]) -- invariant
        dprint("updating row", row)
        self.backing:updateRow(row, self.bufferedGrads[row])
        dprint("after update", row, self.backing:readRow(row))
        self.bufferedGrads[row] = nil
    end
    assert(not self.bufferedGrads[row])
end

local function assertEq(l, r)
    if l ~= r then
        print("assert failure", l, "~=", r)
        assert(l == r)
    end
end

-- Eviction policy
function Way:_chooseVictim()
    local rows = pl.tablex.keys(self.rows)
    -- Random eviction is surprisingly hard to compete with.
    dprint("choosing from", rows)
    assertEq(#rows, self.numRows)
    assert(#rows > 0)
    return rows[torch.random(1, #rows)]
end

-- if row is unspecified, run eviction algorithm
function Way:_evict(row)
    local row = row or self:_chooseVictim()
    assert(self.rows[row])
    if self.rows[row] then
        self:_writeBackOne(row)
        self.rows[row] = nil
        self.numRows = self.numRows - 1
        assert(self.numRows >= 0)
        self:_incStat('evict')
    end
end

function Way:trim()
    while self.numRows >= self.size do
        self:_evict()
    end
end

function Way:flush()
    dprint("flushing a way", self.numRows)
    for row,_ in pairs(self.rows) do
        dprint("flushing row", row)
        self:_evict(row)
    end
    assert(self.numRows == 0)
    self:_incStat('flush')
    dprint("flush done")
end

function Way:pull(row)
    if self.rows[row] then
        self:_incStat('hit')
        return self.rows[row]
    end

    self:_incStat('miss')
    self:trim()
    assert(self.numRows < self.size)
    self.rows[row] = self.backing:readRow(row):clone()
    self.numRows = self.numRows + 1
    assert(not self.bufferedGrads[row])
    return self.rows[row]
end

-- The lookup table itself is a hash table of Ways.
local CachingLookupTable, parent = torch.class('nn.CachingLookupTable',
                                               'nn.Module')

function CachingLookupTable:__init(backingLut, numRows)
    parent.__init(self)
    assert(backingLut)
    assert(numRows)
    assert(numRows <= backingLut.weight:size(1))

    self.backing = backingLut
    self.numRows = numRows
    assert(numRows >= 1)
    self.numCols = backingLut.weight[1]:size(1)
    local perWay = 8
    self.numWays = math.ceil(numRows / perWay)
    dprint("__init(): numWays", self.numWays)
    self.ways = { }
    self.stats = util.defaultdict(function() return 0 end)
    for i = 1,self.numWays do
       table.insert(self.ways, Way(perWay, backingLut, self.stats))
    end
    self.output = torch.Tensor()
end

-- Core mapping function. For now just project; a lot of our use cases have
-- already either frequency-sorted or hashed the row space anyway, either
-- of which is optimal if the cache keyspace is sufficiently smaller than
-- the underlying one.
function CachingLookupTable:_rowToWay(row)
    local idx = 1 + (row % self.numWays)
    assert(idx >= 1)
    assert(idx <= self.numWays)
    dprint("row to way", row, idx)
    return self.ways[idx]
end

function CachingLookupTable:_writeBackOne(row)
    dprint("writing back row", row)
    self:_rowToWay(row):_writeBackOne(row)
end

function CachingLookupTable:flush()
    dprint("CLT:flush flushing cache", self.numWays)
    for i = 1,self.numWays do
        dprint("CLT:flush flushing way", i)
        self.ways[i]:flush()
    end
end

function CachingLookupTable:dumpStats()
    for nm, count in pairs(self.stats) do
        print("STAT", nm, count)
    end
end

function CachingLookupTable:notImplemented(name)
    print("aieee!", name)
    assert(false)
end

function CachingLookupTable:accUpdateOnly()
    -- We don't currently implement any special optimizations for
    -- this case.
    self.backing:accUpdateOnly()
end

function CachingLookupTable:reset(stdv)
    self:flush()
    self.backing:reset(stdv)
end

function CachingLookupTable:readRow(row)
    local way = self:_rowToWay(row)
    return way:pull(row)
end

function CachingLookupTable:writeRow(row, val)
    local lval = self:_rowToWay(row):pull(row)
    lval:copy(val)
end

function CachingLookupTable:updateRow(row, val, lr)
    local way = self:_rowToWay(row)
    way:updateRow(row, val, lr)
end

function CachingLookupTable:updateRows(rows, val)
    self.backing:updateRows(rows, val)
end

function CachingLookupTable:updateOutput(input)
    local function pullOne(row, outputVec)
        local row = self:readRow(row)
        outputVec:copy(row)
    end

    local function updateNRows(inputRows, output)
        for i = 1,input:size(1) do
            pullOne(input[i], output[i])
        end
    end

    if input:dim() == 1 then
        self.output:resize(input:size(1), self.numCols)
        updateNRows(input, self.output)
    else
        assert(input.dim() == 2)
        -- batch, inputs, outputs
        self.output:resize(input:size(1), input:size(2), self.numCols)
        for i = 1,input:size(1) do
            updateNRows(input[i], self.output[i])
        end
    end
    return self.output
end

-- For now we only support the accUpdateGradParameters usage pattern.
function CachingLookupTable:zeroGradParameters()
    self:notImplemented('zeroGradParameters')
end
function CachingLookupTable:accGradParameters()
    self:notImplemented('accGradParameters')
end
function CachingLookupTable:updateParameters(lr)
    self:notImplemented('accGradParameters')
end

function CachingLookupTable:type(type)
    self:flush()
    self.backing:type(type)
    parent.type(self, type)
end

function CachingLookupTable:accUpdateGradParameters(input, gradOutput, lr)
    local lr = lr or 1.0
    local touchedWays = { }
    dprint("accUGP")
    local function getWay(row)
        local way = self:_rowToWay(row)
        dprint("touched way")
        touchedWays[way] = 'yup'
        return way
    end

    local function updateRow(row, addend)
        dprint("writing row ", row)
        getWay(row):updateRow(row, addend, lr)
    end

    local function updateNRows(input, gradOutput)
        dprint("update some rows:", input, input:size(1))
        for i = 1,input:size(1) do
            dprint("update rows:", i, input[i])
            updateRow(input[i], gradOutput[i])
        end
    end

    if input:dim() == 1 then
        dprint("update N rows, dim 1 case")
        updateNRows(input, gradOutput)
    else
        assert(gradOutput:size(1) == input:size(1))
        dprint("update N rows, batch case")
        for i = 1,input:size(1) do
            updateNRows(input[i], gradOutput[i])
        end
    end

    -- return to equilibrium size
    for w,_ in pairs(touchedWays) do
        dprint("trim way")
        w:trim()
    end
end
